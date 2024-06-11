import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import simcem
from simcem.clinker import db, clinkerize
from scipy.optimize import linprog, nnls


import pycalphad
from pycalphad import variables as v
import numpy as np
import matplotlib.pyplot as plt
@st.cache_resource()
def get_wimcem_db():
    return pycalphad.Database('C_A_S_Fe_O_M.tdb')
####

def mole_fractions_to_mass_percentages(mole_fractions, component_molar_masses):
    # Initialize a dictionary to store the mass percentages of components
    mass_percentages = {}

    # Calculate the total mass of the phase
    total_mass = sum(mole_fractions[component] * component_molar_masses[component] for component in mole_fractions)

    # Calculate the mass percentage of each component
    for component, mole_fraction in mole_fractions.items():
        mass = mole_fraction * component_molar_masses[component]
        mass_percentage = (mass / total_mass) * 100
        mass_percentages[component] = mass_percentage

    return mass_percentages

def additional_calculations(dbf, solids):
    L = solids["CaO"] / 56.08
    A = solids["Al2O3"] / 101.96
    Q = solids["SiO2"] / 60.08
    F = solids["Fe2O3"] / 159.69
    S = solids["SO3"] / 80.06
    FE = 2 * F
    O = 3 * F + 3 * S
    total_moles = L + A + Q + FE + O + S
    X_L = L / total_moles
    X_FE = FE / total_moles
    X_A = A / total_moles
    X_Q = Q / total_moles
    X_O = O / total_moles
    X_X = S / total_moles
    
    return X_L, X_A, X_Q, X_X, X_FE, X_O

def M(comp):
    return db.getComponent(comp).mass()

initial_raw_solids = {
    #"limestone": simcem.Components ({'CaCO3':1}),
    #"bauxite": simcem.Components ({'Al₂O₃':0.6932, 'SiO₂':0.1152}),
    #"clay": simcem.Components ({'Al₂O₃':0.3818, 'SiO₂':0.3924}),
    #"Ca4Al6SO16": simcem.Components ({'Ca4Al6SO16':100}),
    "#1 Clay": ({'SiO₂':46.89, 'Al₂O₃':15.85, "Fe₂O₃":6.31, "CaO":9.32, "MgO":3.61, "SO₃":0.01, "Na₂O":0.19, "K₂O":4.75, "TiO₂":0.75, "MnO":0.021, "SrO":0.02, "P₂O₅":0.108}),
    "#1 Grey shale": ({'SiO₂':36.38, 'Al₂O₃':7.12, "Fe₂O₃":3.5, "CaO":20.73, "MgO":6.44, "SO₃":2.44, "Na₂O":0.09, "K₂O":0.43, "TiO₂":0.23, "MnO":0.012, "SrO":0.01, "P₂O₅":0.612}),
    "#1 Brown shale":  ({'SiO₂':55.60, 'Al₂O₃':10.43, "Fe₂O₃":4.07, "CaO":4.82, "MgO":9.31, "SO₃":0.27, "Na₂O":0.32, "K₂O":0.73, "TiO₂":0.41, "MnO":0.008, "SrO":0.01, "P₂O₅":1.724}),
    "#1 Limestone": ({'SiO₂':6.85, 'Al₂O₃':1.26, "Fe₂O₃":0.68, "CaO":50.26, "MgO":2.07, "SO₃":0.29, "Na₂O":0.01, "K₂O":0.22, "TiO₂":0.19, "MnO":0.02, "SrO":0.04, "P₂O₅":0.011}),
    "#1 Iron oxide" : ({"Fe₂O₃":54.05, "SiO₂":4.71, "Al₂O₃":0.99, "CaO":13.75, "MgO":6.17, "SO₃":0.58, "Na₂O":7.24, "K₂O":2.14, "TiO₂":0.25, "Mn₃O₄":2.70, "ZnO":7.17}),
    '"Chinese" Bauxite': {"CaO":0.16, "Al₂O₃":69.32, "SiO₂":11.52, "Fe₂O₃":1.21},
    'Gypsum': ({"SiO₂":1.58, "Al₂O₃":0.06, "Fe₂O₃":0.05, "CaO": 32.95, "MgO": 0.12, "SO₃":57.54}),
    "Anhydrite":  {"CaO":M("CaO") / M("CaSO4") * 100, "SO₃":M("SO3") / M("CaSO4") * 100},
    "Limestone" : {"CaO":M("CaO") / M("CaCO3")*100},
    "Lime" : {"CaO":100},
    "Alumina" : ({"Al₂O₃":100}),
    "Elemental Sulfur" : ({"SO₃":M("SO3")/M("S")*100}),
    "Sillica" : ({"SiO₂":100}),
    # "NMP":({"CaO":2.03, "Al₂O₃":72.93, "Fe₂O₃":1.27, "SiO₂":12.79, "SO₃":0.61}),
    # "Sludge":({"CaO":2.01, "Al₂O₃":70.98, "Fe₂O₃":3.17, "SiO2":22.56, "SO₃":0.01}),
    # "Clay_I":({"CaO":0.06, "Al₂O₃":36.77, "Fe₂O₃":1.96, "SiO2":55.47}),
    # "Clay_R":({"CaO":2.93, "Al₂O₃":19.72, "Fe₂O₃":9.56, "SiO2":56.92, "SO₃":4.83}),
    # '"Gypsum"': ({"SiO₂":2.11, "Al₂O₃":0.74, "Fe₂O₃":0.42, "CaO": 35.82, "SO₃":53.26}),
}
default_include = ["#1 Clay", "#1 Limestone", "Elemental Sulfur", "Gypsum"]
#default_include = ["QNCC_limestone", "Sulfur", "Alumina"]

import collections


excluded_oxides = {"ZnO", "Mn₃O₄", "MnO", "P₂O₅", "SrO"}

def simcem_name(name : str):
    return name.replace("₂", "2").replace("₃", "3").replace("₄", "4").replace("₅", "5")

#Make a list of the oxides used
oxides = list(set(oxide for key, oxides in initial_raw_solids.items() for oxide,amt in oxides.items() if oxide not in excluded_oxides))

initial_targets =collections.defaultdict(float, {
     "alite":0.0,
     "belite":60.0,
     "ye'elimite":30.0,
     "Cement:C4AF":10.0,
     #"CaO",
     #"CaSO4",
     #"SiO2",
})

#st.set_page_config(layout="wide")
st.set_page_config(layout="wide", page_title="Clinker Tool at Bansci.com")
st.title("Clinker Tool")

## Using tabs instead ##
tabs = ["Home", "Upload XRF Data", "Mix Design",'Equilibrium Calculator', "About"]
selected_tab = st.sidebar.selectbox("Select Tab", tabs, index=0)

if selected_tab == "Home":
    st.header("Homepage of the clinker tool")
    st.write("This is the homepage of the clinker tool. In this version, we can compute raw mix designs using a default or you can upload your own XRF for calculations. This tool can also carry out thermodynamic calculations to predict the clinker phases using the calculated designs") 
    st.write("Please select a tool using the sidebar")
    
elif selected_tab == "About":
    st.header("About the tool")
    st.write("This tool was created by Marcus Bannerman with help from Wahab Abdul. It was created to help others access their work on thermodynamics of cement, and uses several technologies. These include simcem and PyCalphad. For any questions please contact Marcus Bannerman.")

@st.cache_resource
def get_avail_IDS():
    alias_to_ID = {alias:key for key, component in db.getComponents().items() for alias in component.getAliases()} | {key: key for key, component in db.getComponents().items()}
    alias_to_ID = {k:v for k, v in alias_to_ID.items() if not k.startswith("NASA") and not k.startswith("InChI") and not k.startswith("PubChem")}
    available_IDs = sorted(db.getComponents().keys())
    return alias_to_ID, available_IDs

alias_to_ID, available_IDs = get_avail_IDS()


wimcem_to_simcem_phases = {'YEL':"ye'elimite",'ALPHA_PRIME':'belite','ALPHA':'belite','ANH':'CaSO4','LIQUID':'Liquid','GAS':'SO2/O2', 'RC3S':'alite', 'FERRITE':'Ferrite','MC3S':'alite','GEH':'gehlinite' }


def formulation_solver():
    df = target_table_df.loc[:]
    #Figure out the molecular mass of each target species
    df['Mol Mass'] = [db.getComponent(alias_to_ID[simcem_name(ID)]).mass() for ID, row in target_table_df.iterrows()]
    #Then calculate the target number of moles
    df['target moles'] = df['Amount'] / df['Mol Mass']
    #Then calculate the target element moles, this is b
    b = target_table_df_moles.set_index('ID').transpose().dot(df['target moles'])

    #Now we need the matrix that converts raw materials into elements
    A = raw_material_table_df_moles.set_index("ID").transpose()

    #Check they line up!
    assert((b.index == A.index).all())
    #Solve the problem
    x, rnorm = nnls(A, b)
    
    #Return the estimate, noting that the raw materials are specified by mass, so have a molar mass of 1
    df = pd.DataFrame({'ID':raw_material_table_df_moles['ID'], 'Mass Amounts':x}).set_index("ID")
    #Normalise the mix
    return df #/ sum(df['Mass Amounts (%)']) * 100

def optimiser(target_rows, target_weights, debug=False):
    #Here we construct the constraint matrix for the targets. All oxides are included to capture any left over items but are not included in the optimisation
    target_df = pd.DataFrame([dict(row) for row in target_rows]+oxide_elements, columns=["ID"] + list(used_elements)).fillna(0)
    #We concat the element matrices for the target and input, and use an
    #index trick to just flip the sign on the numerical columns. We can't
    #preserve the indexing as the inputs and outputs might have the same ID.
    #Add a column used to constrain the total of the input raw materials to 1 mole
    eq_constraint_matrix = pd.concat([target_df,(-1 * raw_material_table_df_moles.set_index("ID")).reset_index()]).reset_index()
    eq_constraint_matrix['TotalConstraint'] = [0.0] * target_df.shape[0] + [1.0] * raw_material_table_df_moles.shape[0]
    n=target_df.shape[0] + raw_material_table_df_moles.shape[0]

    #C is the vector to make the objective function to be minimised. Here we want to max/min the first rows and ignore the rest.
    c = [-w for w in target_weights] + [0]* (n-len(target_weights))

    b_eq = [0.0] * len(used_elements) + [1.0] #Elements cancel, input total sums to 1
    A_eq = eq_constraint_matrix[used_elements + ["TotalConstraint"]].to_numpy().transpose()
    #st.write("Constraints",eq_constraint_matrix)
    #st.write("A_eq", A_eq)
    res = linprog(c, A_eq = A_eq, b_eq = b_eq, bounds=[(0, None)]*n)
    if not res.success:
        st.error("Failed to find max possible amount of " + str(row['ID']))
        st.write(res)
    eq_constraint_matrix["Molar Amounts"] = res.x
    eq_constraint_matrix["Molar Masses"] =  [db.getComponent(alias_to_ID[simcem_name(comp)]).mass() for comp in target_df['ID']] +[1 for comp in raw_material_table_df_moles['ID']]
    eq_constraint_matrix["Mass Amounts"] = eq_constraint_matrix["Molar Amounts"] * eq_constraint_matrix["Molar Masses"]
    #Calculate the mass fractions of the target compounds made
    eq_constraint_matrix["Mass Total"] = [sum(eq_constraint_matrix.iloc[0:target_df.shape[0]]['Mass Amounts'])]*target_df.shape[0] + [sum(eq_constraint_matrix.iloc[target_df.shape[0]:]['Mass Amounts'])]*raw_material_table_df_moles.shape[0]
    eq_constraint_matrix["Mass %"] = 100 *  eq_constraint_matrix["Mass Amounts"] / eq_constraint_matrix["Mass Total"]

    if debug:
        st.write("c")
        st.write(c)
        st.write("b_eq")
        st.write(b_eq)
        st.write("A_eq")
        st.write(A_eq)
        st.write(res)
        st.write("Constraint matrix", eq_constraint_matrix)

    return eq_constraint_matrix


#ID_lookups = { for component in db.getComponents() for alias in component.getAliases()}



#Figure out the elemental composition of all the raw materials
def row_to_elemental(row):
    return dict(sum([row[oxide] / db.getComponent(simcem_name(oxide)).mass() / 100 * db.getComponent(simcem_name(oxide)).getElements() for oxide in oxides], simcem.Components({})))

def thermosolver(df, T_degC, SO2ppm, Equilibrium=False):
    if not Equilibrium:
        # This is for the mode where we have calculated a raw mix design #
        df = df[df['Include'] == True][oxides]
        #Calculate the solid input oxides by mass
        oxides_in_mass = df.transpose().dot(raw_mix_design/100)
        solids_mass = simcem.Components({simcem_name(key):value for key, value in oxides_in_mass['Mass Amounts'].to_dict().items()})
        solids_moles = simcem.MassToMoles(db, solids_mass)
    else:
        # If we want to just calculate using oxides this will be used, the solution is ugly but I wanted it to be in function #
        oxide_percent = {key:value for key, value in zip(df['Oxide'].to_dict().values(), df['Mass Amounts'].to_dict().values())}
        ## Assuming that 100% is 100g i.e mass = percent ##
        solids_mass = simcem.Components(oxide_percent)
        solids_moles = simcem.MassToMoles(db, solids_mass)
    #Convert the SO3 to CaSO4 for insertion
    if solids_moles['CaO'] < solids_moles['SO3']:
        st.error("SO3 in designed raw mix is more than supplied CaO, not sure how to insert it?")
    else:
        solids_moles['CaO'] = solids_moles['CaO'] - solids_moles['SO3']
        solids_moles['CaSO4'] = solids_moles['SO3']
        solids_moles['SO3'] = 0
    solids_moles.removeSmallComponents(1e-16)

    #Now run the solver
    Air = simcem.Components({"O2":21, "N2":79, "CO2":0, "SO2":0, "SO3":0}) / 100
    SO2 = simcem.Components({"SO2":100}) / 100
    gas = Air * (1e6 - SO2ppm) + SO2 * SO2ppm 

    #We use 1M moles of gas as its causing weird effects on the calculation.
    gas, solid, liquid, sys = simcem.clinker.setup_phases(gas, solids_moles, T_degC+273.15)
    sys.equilibrate()
    a = solid.components
    a.removeSmallComponents(1e-7)

    #Now extract the result and compute 
    result = collections.defaultdict(float)
    for key, value in a.items():
        result[key.split(":")[0]] += value *  db.getComponent(key).mass()
    sorted_results = sorted([(v,k) for k,v in result.items()], reverse=True)
    st.write("Predicted stable phases and compositions (using Hanein et al database)")
    results_df = pd.DataFrame([{k:v for v,k in sorted_results}]) 
    st.write(results_df)
    if Equilibrium:
        extra_calcs = {
        'LSF': [(oxide_percent['CaO']) / (2.8 * oxide_percent['SiO2'] + 1.2 * oxide_percent['Al2O3'] + 0.65 * oxide_percent['Fe2O3'])],
        'SR' : [oxide_percent['SiO2'] / (oxide_percent['Al2O3'] + oxide_percent['Fe2O3'])],
        'AR' : [oxide_percent['Al2O3'] / oxide_percent['Fe2O3']],
    }
        extra_calcs_df = pd.DataFrame(extra_calcs)
        st.markdown("**LSF, SR and AR of the oxide raw mix**:")
        st.write("LSF is calculated by CaO/(2.8 SiO2 + 1.2 Al2O3 + 0.65 Fe2O3)")
        st.write(extra_calcs_df)
        return results_df
    st.write(f"Note: Check the converged gas SOx of {(gas.components['SO2']+gas.components['SO3'])/gas.components.N() * 1e6:.0f} PPM is close enough to your setpoint above. Sometimes too much can react.")
    st.markdown("**Oxide compostion of raw mix (for checking on phase diagrams)**")
    oxides_in_mass["Oxide % of raw mix"] = oxides_in_mass["Mass %"]
    oxides_in_mass.sort_values("Oxide % of raw mix", inplace=True, ascending=False)
    oxide_rawmix_df = oxides_in_mass["Oxide % of raw mix"]
    # We have to transpose the above dataframe
    transposed_df = oxide_rawmix_df.transpose()
    ## This is a check for the subscripts in the original names, only applies to the Mix Design page
    if 'SiO₂' in transposed_df.index:
        transposed_df.index= [simcem_name(oxide) for oxide in transposed_df.index]
    ## Calculation of moduli 
    extra_calcs = {
        'LSF': [(transposed_df['CaO']) / (2.8 * transposed_df['SiO2'] + 1.2 * transposed_df['Al2O3'] + 0.65 * transposed_df['Fe2O3'])],
        'SR' : [transposed_df['SiO2'] / (transposed_df['Al2O3'] + transposed_df['Fe2O3'])],
        'AR' : [transposed_df['Al2O3'] / transposed_df['Fe2O3']],
    }
    extra_calcs_df = pd.DataFrame(extra_calcs)
    st.write(oxides_in_mass["Oxide % of raw mix"])
    st.markdown("**LSF, SR and AR of the oxide raw mix**:")
    st.write("LSF is calculated by CaO/(2.8 SiO2 + 1.2 Al2O3 + 0.65 Fe2O3)")
    st.write(extra_calcs_df)
    st.write(f'Total is {oxides_in_mass["Oxide % of raw mix"].sum():.2f}. Note this percentage may be less than expected due to LOI of CO2 not accounted for in the raw mix table.')
    ## I am returning the input solid moles for additional calcs but also a dataframe for the results, this is then saved as a excel sheet 
    return solids_moles, results_df

force_update = False

if "raw_df" not in st.session_state:
    #Setup the initial state of the raw dataframes
    st.session_state['raw_df'] = pd.DataFrame([
        {"ID":k, **{k2:v2 for k2,v2 in v.items() if k2 not in excluded_oxides}, "Include":k in default_include} for k,v in initial_raw_solids.items()
    ], columns = ["ID", "Include"] + sorted(oxides) + ["Total", "CO₂", "Total+CO₂"]).fillna(0)
    st.session_state['target_df'] = pd.DataFrame([
        {"Amount":v} for k,v in initial_targets.items()
    ], columns = ["Amount"], index=[k for k,v in initial_targets.items()]).fillna(0)
    force_update = True

elif selected_tab == 'Equilibrium Calculator':
    st.title('Equilibrium tool')
    st.write('This tab is used to calculate equilibriums for a defined input, this tool uses the Hanein et al database.')
    
    # Initial composition #
    st.header('Input')
    st.write('Insert your input parameters here, any number of oxides are allowed, just double click and add. ')
    solid_composition = {
        'Oxide':['CaO', 'Al2O3', 'SO3', 'Fe2O3', 'SiO2'],
    "Mass Amounts":[52.41, 18.87, 5.99, 3.18, 19.57]
    }
    solid_composition_df = pd.DataFrame(solid_composition)
    oxide_options = ['CaO', 'Al2O3', 'SO3', 'Fe2O3', 'SiO2', 'MgO', 'CaSO4']
    # Make an editable table #
    solid_composition_df= st.data_editor(solid_composition_df,num_rows='dynamic',
        column_config=dict(
        _index=st.column_config.SelectboxColumn("Oxide",required=True,options=oxide_options),
        Amount=st.column_config.NumberColumn("Mass Amount",format="%.2f", min_value=0,))
    )
    # Conditions #
    temperature = st.number_input("Temperature (°C)", value=1450.0,min_value=0.0,step=1.0,max_value=3000.0)
    so2_ppm = st.number_input("SO2ppm", value=0, min_value=0,step=100,max_value=20000)
    if st.button("Calculate Composition"):
        thermosolver(solid_composition_df,T_degC=temperature,SO2ppm=so2_ppm,Equilibrium=True)
## Create a new tab for XRF data upload ##
elif selected_tab == "Upload XRF Data":
    st.title('Upload XRF Data')
    st.header("Example Excel File")
    st.write('This is a examle of the format to be used, you can include or remove as many rows and oxides as you like')
    st.write('You can download the example file below and start filling it in')
    data = {
    'ID': ['Clay', 'Grey Shale', 'Brown Shale' ,'Limestone', 'Iron Oxide', 'Lime'],
    'SiO2': [46.89, 36.38, 55.6, 6.85, 4.71, 0],
    'CaO': [9.32, 20.73, 4.82, 50.26, 13.75, 100.0],
    'Fe2O3': [6.31, 3.5, 4.07, 0.68, 54.05, 0],
    'Al2O3': [15.85, 7.12, 10.43, 1.26, 0.99, 0],
    'SO3': [0.01, 2.44, 0.27, 0.29, 0.58, 0]
    }

    df = pd.DataFrame(data)

# Save the sample dataframe to an Excel file
    df.to_excel('example.xlsx', index=False)
    example_file_path = 'example.xlsx'
    example_df = pd.read_excel(example_file_path)
    st.dataframe(example_df)
    with open(example_file_path, 'rb') as file:
        btn = st.download_button(
            label="Download example Excel file",
            data=file,
            file_name='example.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    debug= st.checkbox("Verbose/Debug mode",False)
    uploaded_file = st.file_uploader("Upload XRF Data (Excel file)", type=["xlsx"])

    if uploaded_file is not None:
        # Read the uploaded Excel file into a DataFrame
        xrf_data = pd.read_excel(uploaded_file)
        xrf_data.insert(0,'Include',True)
        #xrf_data.set_index('ID')
        # Display the XRF data
        st.write("Uploaded XRF Data:")
        st.write(xrf_data)
        df = xrf_data
        df = df.set_index("ID")
        oxides = df.columns[1:].tolist()
    else:
        raise ValueError("Raw material data not found. Please upload raw material data first.")
        
    
    st.header("Target formulation")
    st.write('Specify what you intend to make in mass units. You can add additional compounds by using the empty row at the bottom to search for components.')
    st.write(r'Chemical formulas (i.e, Ca₃SiO₅), mineralogical names (i.e., alite), and cement notation (i.e., Cement\:C3S), can be used to identify a compound, but I don\'t check for duplicates yet!')
    st.write(r'You can delete rows by selecting the row and pressing delete on your keyboard.')
    target_table_df = st.data_editor(st.session_state['target_df'],
                                    num_rows="dynamic",
                                    use_container_width=False,
                                    column_config=dict(
                                        _index=st.column_config.SelectboxColumn("ID",required=True, options=alias_to_ID.keys()),
                                        Amount=st.column_config.NumberColumn("Mass Amount", help="The amount of the phase present", format="%.2f", min_value=0, default=0)
                                    ))
    
    
    raw_material_table_df_moles = pd.DataFrame([{"ID":row['ID']} | row_to_elemental(row) for idx, row in xrf_data.iterrows()])
    #Figure out the elemental composition of the target materials
    target_table_df_moles = pd.DataFrame([{"ID":ID} | dict(db.getComponent(alias_to_ID[ID]).getElements()) for ID, row in target_table_df.iterrows()])
    

    #Determine what elements are in both tables
    used_elements = sorted(set(target_table_df_moles.columns).difference({"ID"}).union(set(raw_material_table_df_moles.columns).difference({"ID"})))

    #Make both tables include all elements, and fill the missing columns with zeros
    raw_material_table_df_moles = raw_material_table_df_moles.reindex(columns=["ID"] + used_elements).fillna(0)
    target_table_df_moles       = target_table_df_moles.reindex(columns=["ID"] + used_elements).fillna(0)

    if debug:
        st.write("Elemental molar compositions of the raw material.")
        st.dataframe(raw_material_table_df_moles, use_container_width=True)
        st.write("Elemental molar compositions for the target phases.")
        st.dataframe(target_table_df_moles, use_container_width=True)

    #For each row in the target table, determine the maximum amount that can be
    #produced. We do this by setting up an optimisation to max the target, while
    #preserving molar balance with the target and all oxides.abs
    # 
    # The variables are the target AND input amounts, subject to limits that the
    #inputs and outputs are positive, and inputs sum to 1.
    #
    #First generate some target rows that correspond to the oxides
    oxide_elements = [{"ID":oxide} | dict(db.getComponent(simcem_name(oxide)).getElements()) for oxide in oxides]
    max_target_data = []
    if debug:
        calc_tabs = st.tabs([row["ID"] for idx, row in target_table_df_moles.iterrows()])
    
    for idx, row in target_table_df_moles.iterrows():
        if debug:
           with calc_tabs[idx]:
                output_df = optimiser([row], [1], debug=debug)
        else:
                output_df = optimiser([row], [1], debug=debug)
        max_target_data.append({"ID": row['ID'], "Max wt%": output_df.iloc[0]["Mass %"]
        } | {row["ID"]:row["Mass %"] for idx,row in output_df.iloc[len(oxides)+1:].iterrows()})
    
    st.header("Formulation limits")
    st.write("Here we calculate what is the maximum possible weight percentage achievable with the raw materials given.")
    st.write("This is to help when designing raw mixes with non-ideal raw materials, to understand what are the maximum limits.")
    st.write("If you use enough analytical/pure raw materials, you should see everything can be achieved at 100% purity according to the mass balance used here.")
    max_target_df = pd.DataFrame(max_target_data, columns=["ID", "Max wt%"] + list(raw_material_table_df_moles['ID']))
    st.dataframe(max_target_df.set_index("ID"), use_container_width=False, column_config={
       row['ID']:st.column_config.NumberColumn(format="%.2f")  for key, row in xrf_data.iterrows()
    } | {
        "Max wt%":st.column_config.NumberColumn(format="%.2f"),
    })



    st.header("Optimised formulation")
    st.write('Here the "closest" raw material formulation to the target formulation is given. Closest means the combination of raw materials that has the least-square difference in elemental composition compared to the target.')
    st.write('If impure raw materials are used, then this "optimal" formulation will not provide the target phases exactly. Even if pure materials are used, thermodynamics/phase stability may also prevent the target phases from appearing. The thermodynamic tools below should be used to check the formulation.')
    st.write('The mass amounts are on the basis of creating the target phases, thus the raw materials may sum to more/less than the target masses given loss on ignition.')
    raw_mix_design = formulation_solver()
    raw_mix_design['Mass %'] =  raw_mix_design['Mass Amounts'] / raw_mix_design['Mass Amounts'].sum() * 100
    st.write(raw_mix_design)

    st.header("Thermodynamic solvers")
    st.write('This thermodynamic solver takes the "optimal" mix calculated above and tries to predict what it will form for particular clinkering conditions.')
    T_degC = st.slider("Clinkering Temperature", 600.0, 1450.0, 1250.0, 1.0, format="%f℃")
    st.write("A limit of 1450℃ for the clinkering temperature is given for OPC but melting is not included in the Hanein et al database so take its results with heavy caution.")
    SO2ppm = st.slider("SO₂ partial pressure", 1.0, 95000.0, 2000.0, 1.0, format="%fPPM")
    st.write("SO₂ partial pressure is limited to 95,000 PPM as that's the max concentration in air to allow full combination to SO₃.")


    solids_moles, results_df = thermosolver(df, T_degC=T_degC, SO2ppm=SO2ppm)
    results_df["T"] = T_degC
    results_df["SO2ppm"] = SO2ppm
    
    results_df["Type"] = "Calculated"
    
    xrf_data["Type"] = "XRF"
    
    ## Saving of results 
    combined_df = pd.concat([xrf_data,results_df])
    
    import io
    buffer=io.BytesIO() 
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        combined_df.to_excel(writer,index=False)
        
    ## Streamlit download button    
    st.download_button(
            label="Download results",
            data=buffer,
            file_name="exported_results.xlsx",
            mime="application/vnd.ms-excel",
        )
        
## This tab is the Mix Design, this might be done away with but is kept in here to prevent Wahab from breaking anything
elif selected_tab == "Mix Design":
    debug= st.checkbox("Verbose/Debug mode",False)
    st.header("Raw Materials")
    st.write("The table below is for any impure raw materials where you have an oxide analysis in weight % (very common in cement where an XRF is used to analyse raw materials).")
    st.write("The total weight column is calculated for you to help you check your data entry. CaO is assumed to be present as CaSO₄ until all SO₃ or CaO present is accounted for, and any remaining CaO is assumed to be present as CaCO₃. This often leads to additional mass (from the CO₂).")
    st.write("You do not need to make the mass total add to 100%, the unknown mass will just be ignored by the calculations, but you must keep an eye on this.")
    st.write("Note: We exclude the following oxides from further analysis due to data limitations", excluded_oxides)
    raw_material_table_df = st.data_editor(st.session_state['raw_df'],
                                            num_rows="dynamic",
                                            use_container_width=False,
                                            column_config=dict(
                                            Total=st.column_config.NumberColumn(disabled=True, format="%.2f"),
                                            Include=st.column_config.CheckboxColumn(default=False),
                                            ) | {
                                              ox :st.column_config.NumberColumn(format="%.2f", default=0.0) for ox in oxides
                                            } | {
                                              "CO₂" : st.column_config.NumberColumn("CO₂ (estimated)", format="%.2f", default=0.0,disabled=True),
                                              "Total+CO₂" : st.column_config.NumberColumn(format="%.2f", default=0.0,disabled=True),
                                            }
                                        )
    if not raw_material_table_df.equals(st.session_state['raw_df']) or force_update:
        df = raw_material_table_df
        df["Total"] = df[oxides].sum(axis=1)
        df["CO₂"] = (df["CaO"] / db.getComponent('CaO').mass() - df["SO₃"] / db.getComponent('SO3').mass()).clip(lower=0) * db.getComponent('CO2').mass()
        df["CO₂"] = df.apply(lambda row: min((100-min(100, max(0, row["Total"]))), row["CO₂"]),axis=1)
        df["Total+CO₂"] = df["Total"] + df["CO₂"]
        st.session_state['raw_df'] = df
        st.experimental_rerun()

    st.header("Target formulation")
    st.write('Specify what you intend to make in mass units. You can add additional compounds by using the empty row at the bottom to search for components.')
    st.write(r'Chemical formulas (i.e, Ca₃SiO₅), mineralogical names (i.e., alite), and cement notation (i.e., Cement\:C3S), can be used to identify a compound, but I don\'t check for duplicates yet!')
    st.write(r'You can delete rows by selecting the row and pressing delete on your keyboard.')
    target_table_df = st.data_editor(st.session_state['target_df'],
                                    num_rows="dynamic",
                                    use_container_width=False,
                                    column_config=dict(
                                        _index=st.column_config.SelectboxColumn("ID",required=True, options=alias_to_ID.keys()),
                                        Amount=st.column_config.NumberColumn("Mass Amount", help="The amount of the phase present", format="%.2f", min_value=0, default=0)
                                    ))

    if not target_table_df.equals(st.session_state['target_df']):
        st.session_state['target_df'] = target_table_df
        st.experimental_rerun()

    
    raw_material_table_df_moles = pd.DataFrame([{"ID":row['ID']} | row_to_elemental(row) for idx, row in raw_material_table_df.iterrows() if row['Include']])
    

    #Figure out the elemental composition of the target materials
    target_table_df_moles = pd.DataFrame([{"ID":ID} | dict(db.getComponent(alias_to_ID[ID]).getElements()) for ID, row in target_table_df.iterrows()])

    #Determine what elements are in both tables
    used_elements = sorted(set(target_table_df_moles.columns).difference({"ID"}).union(set(raw_material_table_df_moles.columns).difference({"ID"})))

    #Make both tables include all elements, and fill the missing columns with zeros
    raw_material_table_df_moles = raw_material_table_df_moles.reindex(columns=["ID"] + used_elements).fillna(0)
    target_table_df_moles       = target_table_df_moles.reindex(columns=["ID"] + used_elements).fillna(0)

    if debug:
        st.write("Elemental molar compositions of the raw material.")
        st.dataframe(raw_material_table_df_moles, use_container_width=True)
        st.write("Elemental molar compositions for the target phases.")
        st.dataframe(target_table_df_moles, use_container_width=True)
        

    #For each row in the target table, determine the maximum amount that can be
    #produced. We do this by setting up an optimisation to max the target, while
    #preserving molar balance with the target and all oxides.abs
    # 
    # The variables are the target AND input amounts, subject to limits that the
    #inputs and outputs are positive, and inputs sum to 1.
    #
    #First generate some target rows that correspond to the oxides
    oxide_elements = [{"ID":oxide} | dict(db.getComponent(simcem_name(oxide)).getElements()) for oxide in oxides]
    max_target_data = []
    if debug:
        calc_tabs = st.tabs([row["ID"] for idx, row in target_table_df_moles.iterrows()])
    


    for idx, row in target_table_df_moles.iterrows():
        if debug:
           with calc_tabs[idx]:
                output_df = optimiser([row], [1], debug=debug)
        else:
                output_df = optimiser([row], [1], debug=debug)
        max_target_data.append({"ID": row['ID'], "Max wt%": output_df.iloc[0]["Mass %"]
        } | {row["ID"]:row["Mass %"] for idx,row in output_df.iloc[len(oxides)+1:].iterrows()})
    
    st.header("Formulation limits")
    st.write("Here we calculate what is the maximum possible weight percentage achievable with the raw materials given.")
    st.write("This is to help when designing raw mixes with non-ideal raw materials, to understand what are the maximum limits.")
    st.write("If you use enough analytical/pure raw materials, you should see everything can be achieved at 100% purity according to the mass balance used here.")
    max_target_df = pd.DataFrame(max_target_data, columns=["ID", "Max wt%"] + list(raw_material_table_df_moles['ID']))
    st.dataframe(max_target_df.set_index("ID"), use_container_width=False, column_config={
       row['ID']:st.column_config.NumberColumn(format="%.2f")  for key, row in raw_material_table_df.iterrows()
    } | {
        "Max wt%":st.column_config.NumberColumn(format="%.2f"),
    })



    st.header("Optimised formulation")
    st.write('Here the "closest" raw material formulation to the target formulation is given. Closest means the combination of raw materials that has the least-square difference in elemental composition compared to the target.')
    st.write('If impure raw materials are used, then this "optimal" formulation will not provide the target phases exactly. Even if pure materials are used, thermodynamics/phase stability may also prevent the target phases from appearing. The thermodynamic tools below should be used to check the formulation.')
    st.write('The mass amounts are on the basis of creating the target phases, thus the raw materials may sum to more/less than the target masses given loss on ignition.')
    raw_mix_design = formulation_solver()
    raw_mix_design['Mass %'] =  raw_mix_design['Mass Amounts'] / raw_mix_design['Mass Amounts'].sum() * 100
    st.write(raw_mix_design)

    ## I am starting to add the modified Bogue and Bogue type calculations here just for comparison
    ABYF = ['alite','belite', 'ye\'elimite', 'Cement:C4AF']
    if all(term in target_table_df.index for term in ABYF):
        st.header("Modified Bogue")
        st.write("This is the bogue output for the above raw oxides, this is just for comparison because people like Bogue type equations. The below thermodynamic solver is far more accurate")
        st.write("The modified bogue is taken from the work of Duvallet et al. 2014")
    
    st.header("Thermodynamic solvers")
    st.write('This thermodynamic solver takes the "optimal" mix calculated above and tries to predict what it will form for particular clinkering conditions.')
    T_degC = st.slider("Clinkering Temperature", 600.0, 1450.0, 1250.0, 1.0, format="%f℃")
    st.write("A limit of 1450℃ for the clinkering temperature is given for OPC but melting is not included in the Hanein et al database so take its results with heavy caution.")
    SO2ppm = st.slider("SO₂ partial pressure", 1.0, 95000.0, 2000.0, 1.0, format="%fPPM")
    st.write("SO₂ partial pressure is limited to 95,000 PPM as that's the max concentration in air to allow full combination to SO₃.")

    df = raw_material_table_df.set_index("ID")
    solids_moles, results_df = thermosolver(df,T_degC=T_degC, SO2ppm=SO2ppm)

    req_oxides = {"CaO":"L", "Al2O3":"A", "SiO2":"Q"}
    comps = collections.defaultdict(float)
    total = 0
    for ox,ox_w in req_oxides.items():
        if ox not in solids_moles:
            comps[ox_w] = 0
        else:
            comps[ox_w] = solids_moles[ox]
            total += solids_moles[ox]
    if "CaSO4" in solids_moles:
        comps["X"] += solids_moles["CaSO4"]
        comps["L"] += solids_moles["CaSO4"]
        comps["O"] += 3 * solids_moles["CaSO4"]
        total += solids_moles["CaSO4"] + solids_moles["CaSO4"] + 3*solids_moles["CaSO4"]
    if "Fe2O3" in solids_moles:
       comps["FE"] += 2 * solids_moles["Fe2O3"]
       comps["O"] += 3 * solids_moles["Fe2O3"]
       total += 2 * solids_moles["Fe2O3"] + 3 * solids_moles["Fe2O3"]
    
    comps = {k:v/total for k,v in comps.items()}
    
    st.header("EXPERIMENTAL: Abdul et al Thermodynamic solver")
    st.write("This solver includes melt phases and high temperature data, but only includes the CaO-SiO2-Al2O3-FeO-Fe2O3-O-S system and is still being developed.")
    st.write("Input solids vector")
    st.write(comps)
    st.write("Simcem solids moles")
    st.write(solids_moles)
    dbf = get_wimcem_db()
    phases = sorted(dbf.phases.keys())
    from pycalphad import equilibrium
    conditions = {#This is temperature
                  v.T:T_degC + 273,
                  # System size (so in this case 1 mole)
                  v.N:1,
                  #Pressure (this is the default_
                  v.P:101325}
    #Skip the first comp, as it is deduced by pycalphad which requires mol frac to sum to one.
    for k in list(comps.keys())[1:]:
        conditions[v.X(k)] = comps[k]

    elements = list(comps.keys())
    if "FE" in comps:
       elements = elements + ["VA"]
    eq = equilibrium(dbf, elements, phases, conditions=conditions, calc_opts ={"pdens":200})
    # st.write("Solver results")
    # st.text(eq)
    st.write("Phases and compositions")
    phase_names_unfiltered = eq['Phase'].squeeze()
    row_selector = (phase_names_unfiltered != '')
    phase_names = phase_names_unfiltered[row_selector]
    fractions = eq["NP"].squeeze()[row_selector]
    if len(phase_names) == 0:
        st.error(f'(Convergence failure) at T='+str(T_degC))
        for k in list(comps.keys())[1:]:
            conditions[v.X(k)] = round(comps[k],3)
        eq = equilibrium(dbf, elements, phases, conditions=conditions, calc_opts ={"pdens":200})
        st.write("Retrying results (I'm rounding off some of the digits in the mole fraction, I found this helps)")
        st.write("Phases and compositions (Ph %)")
        phase_names_unfiltered = eq['Phase'].squeeze()
        row_selector = (phase_names_unfiltered != '')
        phase_names = phase_names_unfiltered[row_selector]
        fractions = eq["NP"].squeeze()[row_selector]
    results_dict = {}
    for ph,fr in zip(phase_names.values,fractions.values):
        x = wimcem_to_simcem_phases[ph]
        results_dict[x] = [fr*100]
    results_df = pd.DataFrame(results_dict)
    st.write(results_df)
