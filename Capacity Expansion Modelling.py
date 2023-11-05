import os
import pypsa
glpk_path = "/usr/local/bin/glpsol"
os.environ["PATH"] += os.pathsep + os.path.dirname(glpk_path) # the only way I got glpsol working...
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("bmh")

# set up to pull pypsa database's assumptions and projections
# use same assumptions given in example problem

year = 2030
url = f"https://raw.githubusercontent.com/PyPSA/technology-data/master/outputs/costs_{year}.csv"
costs = pd.read_csv(url, index_col=[0, 1])

costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
costs.unit = costs.unit.str.replace("/kW", "/MW")

defaults = {
    "FOM": 0,
    "VOM": 0,
    "efficiency": 1,
    "fuel": 0,
    "investment": 0,
    "lifetime": 25,
    "CO2 intensity": 0,
    "discount rate": 0.07,
}
costs = costs.value.unstack().fillna(defaults)

costs.at["OCGT", "fuel"] = costs.at["gas", "fuel"]
costs.at["CCGT", "fuel"] = costs.at["gas", "fuel"]
costs.at["OCGT", "CO2 intensity"] = costs.at["gas", "CO2 intensity"]
costs.at["CCGT", "CO2 intensity"] = costs.at["gas", "CO2 intensity"]

# annuity formula

def annuity(r, n):
    return r / (1.0 - 1.0 / (1.0 + r) ** n)

annuity(0.07, 20)

# STMGC

costs["marginal_cost"] = costs["VOM"] + costs["fuel"] / costs["efficiency"]

# annualized investment cost

annuity = costs.apply(lambda x: annuity(x["discount rate"], x["lifetime"]), axis=1)
costs["capital_cost"] = (annuity + costs["FOM"] / 100) * costs["investment"]

# pull wind, solar load time series data

url = (
    "https://tubcloud.tu-berlin.de/s/pKttFadrbTKSJKF/download/time-series-lecture-2.csv"
)
ts = pd.read_csv(url, index_col=0, parse_dates=True)
ts.head(3)

# convert MW to GW
ts.load *= 1e3

# sample every other hour
resolution = 4
ts = ts.resample(f"{resolution}H").first()

# create network

n = pypsa.Network()
n.add("Bus", "electricity")
n.set_snapshots(ts.index)
n.snapshots

n.snapshot_weightings.head(3)
n.snapshot_weightings.loc[:, :] = resolution
n.snapshot_weightings.head(3)

carriers = [
    "onwind",
    "offwind",
    "solar",
    "OCGT",
    "hydrogen storage underground",
    "battery storage",
]

n.madd(
    "Carrier",
    carriers,
    color=["dodgerblue", "aquamarine", "gold", "indianred", "magenta", "yellowgreen"],
    co2_emissions=[costs.at[c, "CO2 intensity"] for c in carriers],
)

n.add(
    "Load",
    "demand",
    bus="electricity",
    p_set=ts.load,
)

fig1, ax1 = plt.subplots(figsize=(6, 2))
n.loads_t.p_set.plot(ax=ax1, ylabel="MW")
ax1.set_title('MW over time')

n.add(
    "Generator",
    "OCGT",
    bus="electricity",
    carrier="OCGT",
    capital_cost=costs.at["OCGT", "capital_cost"],
    marginal_cost=costs.at["OCGT", "marginal_cost"],
    efficiency=costs.at["OCGT", "efficiency"],
    p_nom_extendable=True,
)

for tech in ["onwind", "offwind", "solar"]:
    n.add(
        "Generator",
        tech,
        bus="electricity",
        carrier=tech,
        p_max_pu=ts[tech],
        capital_cost=costs.at[tech, "capital_cost"],
        marginal_cost=costs.at[tech, "marginal_cost"],
        efficiency=costs.at[tech, "efficiency"],
        p_nom_extendable=True,
    )

fig2, ax2 = plt.subplots(figsize=(6, 2))
n.generators_t.p_max_pu.loc["2015-05"].plot(ax=ax2, ylabel="CF")
ax2.set_title('Capacity Factor in May 2015')

plt.show()

n.optimize(solver_name="glpk")

# as per https://pypsa.readthedocs.io/en/latest/examples/statistics.html get stats on solved model


# model evaluation

n.objective / 1e9

n.generators.p_nom_opt.div(1e3)  # GW

n.snapshot_weightings.generators @ n.generators_t.p.div(1e6)  # TWh

opex = n.snapshot_weightings.generators @ (
    n.generators_t.p * n.generators.marginal_cost
).div(
    1e6
)  # M€/a

capex = (n.generators.p_nom_opt * n.generators.capital_cost).div(1e6)  # M€/a
capex + opex

(n.statistics.capex() + n.statistics.opex(aggregate_time="sum")).div(1e6)

emissions = (
    n.generators_t.p
    / n.generators.efficiency
    * n.generators.carrier.map(n.carriers.co2_emissions)
)  # t/h

n.snapshot_weightings.generators @ emissions.sum(axis=1).div(1e6)  # Mt

# total emissions

emissions = (
    n.generators_t.p
    / n.generators.efficiency
    * n.generators.carrier.map(n.carriers.co2_emissions)
)  # t/h

n.snapshot_weightings.generators @ emissions.sum(axis=1).div(1e6)  # Mt


# plotting optimal dispatch

def plot_dispatch(n, time="2015-07"):
    p_by_carrier = n.generators_t.p.groupby(n.generators.carrier, axis=1).sum().div(1e3)

    if not n.storage_units.empty:
        sto = (
            n.storage_units_t.p.groupby(n.storage_units.carrier, axis=1).sum().div(1e3)
        )
        p_by_carrier = pd.concat([p_by_carrier, sto], axis=1)

    fig3, ax = plt.subplots(figsize=(6, 3))

    color = p_by_carrier.columns.map(n.carriers.color)

    p_by_carrier.where(p_by_carrier > 0).loc[time].plot.area(
        ax=ax,
        linewidth=0,
        color=color,
    )

    charge = p_by_carrier.where(p_by_carrier < 0).dropna(how="all", axis=1).loc[time]

    if not charge.empty:
        charge.plot.area(
            ax=ax,
            linewidth=0,
            color=charge.columns.map(n.carriers.color),
        )

    n.loads_t.p_set.sum(axis=1).loc[time].div(1e3).plot(ax=ax, c="k")

    plt.legend(loc=(1.05, 0))
    ax.set_ylabel("GW")
    ax.set_ylim(-200, 200)

plot_dispatch(n)
plt.show()


print("Objective function value:", n.objective)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print("Optimal installed capacities:", n.generators.p_nom_opt)
print("overview", n.statistics().dropna())


# MC of each generator
for gen in n.generators.index:
    print(f"Generator: {gen}, Marginal Cost: {n.generators.loc[gen, 'marginal_cost']}")

# find the hour with highest % of load sat'd by solar and OCGT\
# Calculate the percentage of the load met by each generator type
gen_output = n.generators_t.p.divide(n.loads_t.p_set['demand'], axis=0)

# Find the hour with the highest solar contribution
max_solar_hour = gen_output['solar'].idxmax()

# Find the hour with the highest OCGT contribution
max_ocgt_hour = gen_output['OCGT'].idxmax()

# print max values
max_solar_value = gen_output.loc[max_solar_hour, 'solar']  # getting the max value for solar
max_ocgt_value = gen_output.loc[max_ocgt_hour, 'OCGT']  # getting the max value for OCGT

print(f"The hour with the highest solar contribution is {max_solar_hour} with a value of {max_solar_value:.2%}")
print(f"The hour with the highest OCGT contribution is {max_ocgt_hour} with a value of {max_ocgt_value:.2%}")

# load at max solar
load_at_max_solar_hour = n.loads_t.p_set.loc[max_solar_hour, 'demand']

# load at max OCGT
load_at_max_ocgt_hour = n.loads_t.p_set.loc[max_ocgt_hour, 'demand']

print(f"The load at the hour with the highest solar contribution ({max_solar_hour}) is {load_at_max_solar_hour:.2f} MW.")
print(f"The load at the hour with the highest OCGT contribution ({max_ocgt_hour}) is {load_at_max_ocgt_hour:.2f} MW.")

# cost at max
solar_MC = n.generators.loc['solar','marginal_cost']
solar_cost_at_max = solar_MC * load_at_max_solar_hour

ocgt_MC = n.generators.loc['OCGT','marginal_cost'] #mc
ocgt_cost_at_max = ocgt_MC * load_at_max_ocgt_hour

print(f"The cost of electricity at solar max, {max_solar_hour} is €{solar_cost_at_max:.2f}")
print(f"The cost of electricity at ocgt max, {max_solar_hour} is €{ocgt_cost_at_max:.2f}")

# find an hour with 50% solar contribution
# Get the maximum solar generation value
max_solar_value = gen_output['solar'].max()

# Halve it - we know this because we already found a 100% solar hour in this system
target_solar_value = max_solar_value / 2

# small enough tolerance for 1 result (found this manually)
tolerance = max_solar_value * 0.0005

# Find the hour(s) where the solar generation is within the tolerance of the 50% target
solar_50_percent_hours = gen_output['solar'][
    (gen_output['solar'] >= target_solar_value - tolerance) &
    (gen_output['solar'] <= target_solar_value + tolerance)
]

# Assuming `solar_50_percent_hours` has only one hour after adjusting the tolerance
if len(solar_50_percent_hours) == 1:
    # Get the hour index for the 50% solar generation
    hour_50_percent_solar = solar_50_percent_hours.idxmax()

    # fractional data for all generators at this hour
    fractional_output_at_50_percent_solar = gen_output.loc[hour_50_percent_solar]

    # Display the fractional output for each generator during this hour
    print(f"Fractional output for each generator during the hour with approximately 50% solar generation (Hour: {hour_50_percent_solar}):")
    print(fractional_output_at_50_percent_solar)
else:
    print("The tolerance adjustment did not result in a single hour")

# load at 50% solar

load_at_50_solar_hour = n.loads_t.p_set.loc[hour_50_percent_solar, 'demand']
print(f"The load at the hour with 50% solar contribution ({hour_50_percent_solar}) is {load_at_50_solar_hour:.2f} MW.")

MW_output_at_50_percent_solar = fractional_output_at_50_percent_solar * load_at_50_solar_hour

print(f"Output of each generator at the hour with 50% solar contribution (Hour: {hour_50_percent_solar})")
print(MW_output_at_50_percent_solar)

# electricity price at 50% solar hour
# was having trouble pulling the MCs from earlier in the code, so I declared them manually

marginal_costs = {
    'OCGT': 53.524390243902445,
    'onwind': 1.35,
    'offwind': 0.02,
    'solar': 0.01
}

cost_at_50_percent_solar = MW_output_at_50_percent_solar.multiply(pd.Series(marginal_costs))


print(f"Cost for each generator during the hour with approximately 50% solar generation (Hour: {hour_50_percent_solar}):")
print(cost_at_50_percent_solar)

# take sum
total_cost_at_50_percent_solar = cost_at_50_percent_solar.sum()
print(f"Total cost at the hour with 50% solar contribution ({hour_50_percent_solar}) is €{total_cost_at_50_percent_solar:.2f}")


# plot supply stacks

# 100% solar hour
# need to adjust bar width for volume

supply_stack = pd.DataFrame({
    'Generator': n.generators.index,
    'Marginal Cost': n.generators['marginal_cost'],
    'Capacity (MW)': n.generators['p_nom_opt']
})

# sort by MC
supply_stack = supply_stack.sort_values(by='Marginal Cost')

# plot
fig, ax = plt.subplots(figsize=(10, 6))
supply_stack.plot(kind='bar', x='Generator', y='Marginal Cost', ax=ax,
                  color=[n.carriers.loc[gen, 'color'] for gen in supply_stack['Generator']],
                  legend=False)

ax.set_ylabel('Marginal Cost (€/MWh)')
ax.set_xlabel('Generator')
ax.set_title('Supply Stack')

plt.tight_layout()
plt.show()

