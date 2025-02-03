import streamlit as st
import pandas as pd
import plotly.express as px

from datetime import datetime, timedelta

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


st.set_page_config(layout="wide")

st.title('Drop Point Consolidation Tool')

def store_classification(df,filter_category,threshold):
    df_filtered=df[df['customer cluster']==filter_category]
#     df_grouped=df_filtered.groupby(['drop_point','DC']).agg({'TOTPAL':'sum','km':'mode'}).reset_index().sort_values(by='TOTPAL')
    df_grouped = df_filtered.groupby(['drop_point', 'DC','customer cluster']).agg({'TOTPAL': 'sum', 'km': lambda x: pd.Series.mode(x)[0]}).reset_index().sort_values(by='TOTPAL')

    df_grouped['cumsum']=df_grouped['TOTPAL'].cumsum()
    df_grouped['cumsum_percentage']=100*(df_grouped['cumsum']/df_grouped['TOTPAL'].sum())
    df_grouped['Tier']=np.where(df_grouped['cumsum_percentage']<=threshold,'Tier2','Tier1')
    return df_grouped


def compute_shipment_cost(pallets, distance, rate_card_df, max_pallets=33):
    # Round up the distance
    distance = int(np.ceil(distance))
    full_trucks, partial_trucks = divmod(pallets, max_pallets)
    # Cost for full trucks
    full_truck_cost = 0
    if full_trucks > 0:
        # Get rate 
        rate = rate_card_df.loc[(rate_card_df['Distance (LL)'] <= distance) & (rate_card_df['Distance (UL)'] >= distance), max_pallets].iloc[0]
        # Get cost
        full_truck_cost = full_trucks * (rate * max_pallets)
    # Cost for partial trucks
    partial_truck_cost = 0
    if partial_trucks > 0:
        # Get rate 
        rate = rate_card_df.loc[(rate_card_df['Distance (LL)'] <= distance) & (rate_card_df['Distance (UL)'] >= distance), partial_trucks].iloc[0]
        # Get cost
        partial_truck_cost = rate * partial_trucks
    return full_truck_cost + partial_truck_cost



def allocate_tier2_to_tier1(df_orig, df_new, df_road_dist, max_distance_threshold):
    """
    Allocates each Tier2 drop point (df_orig) to exactly one Tier1 drop point (df_new), 
    ensuring that:
    1. They are served by the same warehouse (DC).
    2. Tier1 is closer to DC than Tier2.
    3. Tier2 → Tier1 distance is within max_distance_threshold.
    4. Tier1 has enough capacity to accommodate Tier2’s pallets.

    Args:
        df_orig (pd.DataFrame): Tier2 drop points.
        df_new (pd.DataFrame): Tier1 drop points.
        df_road_dist (pd.DataFrame): Distance between drop points.
        max_distance_threshold (float): Max allowed Tier2 → Tier1 distance.

    Returns:
        pd.DataFrame: Tier2 to Tier1 allocation results.
    """

    # Initialize results list
    allocations = []

    # Track remaining capacity for Tier1 drop points
    df_new = df_new.copy()  # Prevent modifying original DataFrame
    df_new['remaining_qty'] = df_new['allowed_qty']

    # Iterate through each Tier2 drop point
    for _, tier2_row in df_orig.iterrows():
        tier2_point = tier2_row['drop_point']
        tier2_postal = tier2_row['postal_code']
        tier2_dc = tier2_row['DC']
        tier2_distance = tier2_row['Km mode']
        tier2_pallets = tier2_row['TOTPAL']

        # Filter Tier1 drop points: Same DC, Closer to DC, and Has Capacity
        candidates = df_new[
            (df_new['DC'] == tier2_dc) &
            (df_new['Km mode'] < tier2_distance) &
            (df_new['remaining_qty'] > 0)
        ].copy()

        if candidates.empty:
            continue  # No valid Tier1 candidates, skip this Tier2 point

        # Merge with df_road_dist to check distance constraint
        candidates = candidates.merge(df_road_dist, 
                                      left_on='postal_code', 
                                      right_on='dest_postal_code', 
                                      how='left', indicator=True)

        # Debug: Check if merging caused unexpected data loss
        print(f"Merging distances for Tier2 {tier2_point}: {candidates['_merge'].value_counts()}")

        # Ensure only valid Tier2-Tier1 distances are considered
        candidates = candidates[
            (candidates['orig_postal_code'] == tier2_postal) &  # Match Tier2's postal code
            (candidates['distance'] <= max_distance_threshold)  # Distance constraint
        ]

        # Sort by distance to Tier2 first, then distance to DC
        candidates = candidates.sort_values(by=['distance', 'Km mode'])
#         candidates = candidates.sort_values(by=['Km mode','distance' ])


        # Pick the **closest valid Tier1** drop point
        if not candidates.empty:
            best_tier1 = candidates.iloc[0]
            tier1_point = best_tier1['drop_point']
            available_qty = best_tier1['remaining_qty']

            # Debugging before allocation
            print(f"Allocating {tier2_point} to {tier1_point}:")
            print(f" - Distance to Tier1: {best_tier1['distance']} km")
            print(f" - Distance to DC (Tier2): {tier2_distance} km")
            print(f" - Distance to DC (Tier1): {best_tier1['Km mode']} km")
            print(f" - Available capacity: {available_qty}")
            print(f" - Required pallets: {tier2_pallets}")

            # Check if Tier1 can accommodate the Tier2 store's full pallet quantity
            if tier2_pallets <= available_qty:
                # Allocate Tier2 to Tier1
                allocations.append({
                    'Tier2_drop_point': tier2_point,
                    'Tier2_postal_code': tier2_postal,
                    'Tier2_DC': tier2_dc,
                    'Tier1_drop_point': tier1_point,
                    'Tier1_postal_code': best_tier1['postal_code'],
                    'Tier1_DC': best_tier1['DC'],
                    'Distance_to_Tier1': best_tier1['distance'],
                    'Original_Distance_to_DC': tier2_distance,
                    'New_Distance_to_DC': best_tier1['Km mode'],
                    'Allocated_qty': tier2_pallets
                })

                # Update the Tier1 store’s remaining capacity
                df_new.loc[df_new['drop_point'] == best_tier1['drop_point'], 'remaining_qty'] -= tier2_pallets

    # Convert to DataFrame
    return pd.DataFrame(allocations)

def plot_movement_map(df):
    """
    Creates a Plotly Mapbox visualization of movements from origin to destination.
    
    Parameters:
    df (pd.DataFrame): A DataFrame containing the following columns:
        - orig_city (str): Name of the origin city
        - orig_lat (float): Latitude of the origin city
        - orig_long (float): Longitude of the origin city
        - dest_city (str): Name of the destination city
        - dest_lat (float): Latitude of the destination city
        - dest_long (float): Longitude of the destination city
        
    Returns:
    fig (plotly.graph_objects.Figure): The generated movement map.
    """
    
    # Create a scatter map of origin points with text labels
    fig = px.scatter_mapbox(df,
                            lat="orig_lat",
                            lon="orig_long",
                            hover_name="City Original",  # Show origin city on hover
                            text="City Original",  # Display origin city name on the marker
                            zoom=5,
                            mapbox_style="carto-darkmatter")

    # Add destination points with text labels
    fig.add_trace(
        go.Scattermapbox(
            lat=df["dest_lat"],
            lon=df["dest_long"],
            mode="markers+text",
            marker=dict(size=10, color="Green"),
            text=df["City New"],  # Display destination city name on the marker
            textposition="top center",  # Position the text above the marker
            hoverinfo="text",
            hovertext=df["City New"],  # Show destination city on hover
            name="Destination"
        )
    )

    # Add arrow lines for movement
    for _, row in df.iterrows():
        fig.add_trace(
            go.Scattermapbox(
                mode="lines+markers",
                lon=[row["orig_long"], row["dest_long"]],
                lat=[row["orig_lat"], row["dest_lat"]],
                line=dict(width=2, color="Red"),
                hoverinfo="text",
                 text=f"{row['Customer Original']} ({row['Street Original']}) → {row['Customer New']} ({row['Street New']})",  # Show movement as text
                name="Movement",
                marker=dict(size=10, symbol="arrow",  color="Red")  # Add arrow marker
            )
        )

    # Update layout to center the map on Germany
    fig.update_layout(
        title="Movements from Origin to Destination (Germany)",
        mapbox=dict(
            center=dict(lat=51.1657, lon=10.4515),
            zoom=5
        ),
        showlegend=True
    )

    return fig






# @st.cache_data
def load_cust_data():
    """Load and cache the original dataset."""
    return pd.read_excel('customer master data with cluster information.xlsx')

# @st.cache_data
def load_order_data():
    """Load and cache the original dataset."""
    return pd.read_csv('df_tr_pr_&_dhl_with_cust_no_23.csv',parse_dates=['Lst.datum'])

# @st.cache_data
def load_dist_data():
    return pd.read_csv('road_distance_between_zipcodes(5).csv')

# @st.cache_data
def load_rate_card_df():
    return pd.read_excel('Rate Card.xlsx')

rate_card_df = load_rate_card_df()
df_road_dist=load_dist_data()
df_cust=load_cust_data()
df_final_all=load_order_data()


df_cust=df_cust.drop_duplicates(subset='Debitor')
df_cust['customer cluster'] = df_cust['customer cluster'].replace('Sales rep', 'Sales Rep')
df_cust['customer cluster'] = df_cust['customer cluster'].replace('grocery', 'Grocery')
# df_cust['drop_point']=df_cust['postal code'].astype(str)+'_'+df_cust['Name 1']+'_'+df_cust['street']



df_final_all['postal code']=df_final_all['postal code'].astype(int)
df_final_all['postal code']=df_final_all['postal code'].astype(str)
df_final_all['postal code']=df_final_all['postal code'].str.strip()

df_final_all['customer']=df_final_all['customer'].astype(str)
df_final_all['customer']=df_final_all['customer'].str.lower()
df_final_all['customer']=df_final_all['customer'].str.strip()

df=pd.merge(df_final_all,df_cust,left_on='Kundennummer',right_on='Debitor',how='left')

df_fil=df[df['Debitor'].notnull()]
df_fil['drop_point']=df_fil['postal code'].astype(str)+'_'+df_fil['Name 1']+'_'+df_fil['street']
postal_code_city=df_cust.groupby(['postel code','location']).size().reset_index()
postal_code_city=postal_code_city.drop_duplicates(subset=['postel code'])
postal_code_city=postal_code_city[['postel code','location']]



st.sidebar.header("Number Inputs")

num1 = st.sidebar.number_input("Wholesale:", value=5,min_value=0)  # Default value
num2 = st.sidebar.number_input("Grocery:", value=5,min_value=0)
num3 = st.sidebar.number_input("C&C:", value=10,min_value=0)

Max_qty_factor = st.sidebar.number_input("Max_qty_factor:", value=1,min_value=0)
dist_threshold = st.sidebar.number_input("Max_dist_between_small_and_big_store:", value=100,min_value=0)

# Run button in the main app
run_button = st.sidebar.button("Run")

if run_button:
    st.write("Running with the following numbers:")
    if num1==0:
        st.write(f"1. Wholesale: Not consolidating any Wholesale Store.")
    else:
        st.write(f"1. Wholesale: Consolidating Wholesale stores contributing for bottom {num1}% demand (pallets).")
        
    if num2==0:
        st.write(f"2. Grocery: Not consolidating any Grocery Store.")
    else:
        st.write(f"2. Grocery: Consolidating Grocery stores contributing for bottom {num2}% demand (pallets).")
        
    if num3==0:
        st.write(f"3. C&C:Not consolidating any C&C Store.")
    else:    
        st.write(f"3. C&C:Consolidating C&C stores contributing for bottom {num3}% demand (pallets).")
    st.write(f"4. The new store can take maximum upto {Max_qty_factor} times of its own demand (pallets).")
    st.write(f"5. The distance between Original and New store should not exceed {dist_threshold} KM.")

    df_wholesale=store_classification(df_fil,'Wholesale',num1)
    df_grocery=store_classification(df_fil,'Grocery',num2)
    df_cc=store_classification(df_fil,'C&C',num3)


    df_cust_class=pd.concat([df_wholesale,df_grocery,df_cc])
    df_cust_class.rename(columns={'km':'Km mode'},inplace=True)
    # df_cust_class



    df_orig=df_cust_class[(df_cust_class['Tier']=='Tier2')].sort_values(by=['TOTPAL','Km mode'],ascending=[False,True])
    df_orig['postal_code'] = df_orig['drop_point'].str.split('_', n=1, expand=True)[0]
    df_orig['postal_code'] =df_orig['postal_code'] .astype(int)
    
    

    ## filter the Tier-1 wholesale stores, these will serve as potential new drop points.
    df_new=df_cust_class[(df_cust_class['Tier']=='Tier1') & (df_cust_class['customer cluster']=='Wholesale')].sort_values(by=['Km mode','TOTPAL'],ascending=[True,False])
    df_new['postal_code'] = df_new['drop_point'].str.split('_', n=1, expand=True)[0]
    df_new['postal_code']=df_new['postal_code'].astype(int)


    df_new['%']=Max_qty_factor
    df_new['allowed_qty']=round(df_new['TOTPAL']*df_new['%'],0)


    allocation_df=allocate_tier2_to_tier1(df_orig, df_new, df_road_dist, max_distance_threshold=dist_threshold)
    allocation_df=allocation_df.drop_duplicates(subset=['Tier2_drop_point'],keep='first')
    
    allocation_df_with_dist=pd.merge(allocation_df,df_new[['drop_point','DC','Km mode']],left_on=['Tier1_drop_point','Tier1_DC'],right_on=['drop_point','DC'],how='left')
    

    #########
    # Total cost Before
    tour_Id_group_km_before=df_fil.groupby(['Tour-Nr.','km','drop_point','DC'])[['TOTPAL','Betrag']].sum().reset_index()

    tour_Id_group_km_before['total_cost'] = tour_Id_group_km_before[['TOTPAL','km']].apply(lambda x: compute_shipment_cost(x[0], x[1], rate_card_df, 33), axis=1)

    #######
    #total cost after
    df_fil_j=pd.merge(df_fil,allocation_df_with_dist[['Tier2_drop_point','Tier2_DC','Tier1_drop_point','Km mode']],left_on=['drop_point','DC'],right_on=['Tier2_drop_point','Tier2_DC'],how='left')

    df_fil_j['drop_point_final']=np.where(df_fil_j['Tier1_drop_point'].notnull(),df_fil_j['Tier1_drop_point'],df_fil_j['drop_point'])
    df_fil_j['km_final']=np.where(df_fil_j['Km mode'].notnull(),df_fil_j['Km mode'],df_fil_j['km'])

    tour_Id_group_km_after=df_fil_j.groupby(['Tour-Nr.','km_final','drop_point_final','DC'])[['TOTPAL','Betrag']].sum().reset_index()
    tour_Id_group_km_after['total_cost'] = tour_Id_group_km_after[['TOTPAL','km_final']].apply(lambda x: compute_shipment_cost(x[0], x[1], rate_card_df, 33), axis=1)

    original_cost=round(tour_Id_group_km_before['total_cost'].sum(),0)
    updated_cost=round(tour_Id_group_km_after['total_cost'].sum(),0)
    savings=original_cost-updated_cost

    df_space_req=pd.merge(allocation_df_with_dist.groupby(['Tier1_drop_point','Tier1_DC'])['Allocated_qty'].sum().reset_index(),df_new[['drop_point','DC','TOTPAL']],left_on=['Tier1_drop_point','Tier1_DC'],right_on=['drop_point','DC'])
    df_space_req['Increase In Pallets (percentage)']=round(100*(df_space_req['Allocated_qty']/df_space_req['TOTPAL']),2)
    df_space_req[['Postal Code New', 'Customer New', 'Street New']] = df_space_req['Tier1_drop_point'].str.split('_', expand=True)
    df_space_req.rename(columns={'TOTPAL':'Own Demand (Pallets)','Allocated_qty':'Allocated_qty (Pallets)','DC':'Serving Warehouse'},inplace=True)
    df_space_req=df_space_req[['Customer New','Street New','Postal Code New','Serving Warehouse','Own Demand (Pallets)','Allocated_qty (Pallets)','Increase In Pallets (percentage)']]
    
    
    allocation_data_cleaned=allocation_df.copy()
    df_orig[['postal code', 'customer', 'street']] = df_orig['drop_point'].str.split('_', expand=True)
    
    allocation_data_cleaned[['Postal Code original', 'Customer Original', 'Street Original']] = allocation_data_cleaned['Tier2_drop_point'].str.split('_', expand=True)
    allocation_data_cleaned['Postal Code original']=allocation_data_cleaned['Postal Code original'].astype(str)


    allocation_data_cleaned[['Postal Code New', 'Customer New', 'Street New']] = allocation_data_cleaned['Tier1_drop_point'].str.split('_', expand=True)
    allocation_data_cleaned['Postal Code New']=allocation_data_cleaned['Postal Code New'].astype(str)

    allocation_data_cleaned['Postal Code original int']=allocation_data_cleaned['Postal Code original'].astype(int)
    allocation_data_cleaned['Postal Code New int']=allocation_data_cleaned['Postal Code New'].astype(int)

    original_city=postal_code_city.copy()
    new_city=postal_code_city.copy()

    original_city.rename(columns={"location":'City Original'},inplace=True)
    new_city.rename(columns={"location":'City New'},inplace=True)

    a=df_road_dist.groupby(['orig_postal_code','orig_lat','orig_long']).size().reset_index()
    a=a[['orig_postal_code','orig_lat','orig_long']]


    b=df_road_dist.groupby(['dest_postal_code','dest_lat','dest_long']).size().reset_index()
    b=b[['dest_postal_code','dest_lat','dest_long']]




    allocation_data_cleaned=pd.merge(allocation_data_cleaned,df_orig[['postal code','customer','street','customer cluster']],left_on=['Postal Code original', 'Customer Original', 'Street Original'],right_on=['postal code','customer','street'],how='left')
    allocation_data_cleaned=pd.merge(allocation_data_cleaned,original_city,left_on=['Postal Code original int'],right_on='postel code',how='left')
    allocation_data_cleaned=pd.merge(allocation_data_cleaned,new_city,left_on=['Postal Code New int'],right_on='postel code',how='left')
    allocation_data_cleaned=pd.merge(allocation_data_cleaned,a,left_on=['Postal Code original int'],right_on=['orig_postal_code'],how='left')
    allocation_data_cleaned=pd.merge(allocation_data_cleaned,b,left_on=['Postal Code New int'],right_on=['dest_postal_code'],how='left')


    allocation_data_cleaned.rename(columns={'Tier2_DC':'original DC','Tier1_DC':'New DC','Distance_to_Tier1':'Distance (Original to New DP)','Original_Distance_to_DC':'Distance (DC to Orig DP)','New_Distance_to_DC':'Distance (DC to New DP)','customer cluster':'Customer Cluster Original'},inplace=True)
    allocation_data_cleaned['Customer Cluster New']='Wholesale'



    allocation_data_cleaned_filt=allocation_data_cleaned[['Customer Original','Postal Code original','City Original' , 'Street Original','Customer Cluster Original','Customer New','Postal Code New','City New', 'Street New','Customer Cluster New','original DC','New DC','Distance (Original to New DP)','Distance (DC to Orig DP)','Distance (DC to New DP)','Allocated_qty']]
    allocation_data_cleaned_filt=allocation_data_cleaned_filt.drop_duplicates()
    
    
    col1, col2,col3 = st.columns(3)
    col1.metric("Total cost original (2023)", f"€{original_cost:,.2f}")
    col2.metric("Total cost updated (2023)", f"€{updated_cost:,.2f}")
    col3.metric("Total Savings (2023)", f"€{savings:,.2f}")
    
    st.header('Allocation Data')
    st.dataframe(allocation_data_cleaned_filt)
    st.header('Profile of Percentage Increase in Pallets (New Stores)')
    st.dataframe(df_space_req)
    
    fig = plot_movement_map(allocation_data_cleaned)
    st.plotly_chart(fig, use_container_width=True)
    #
