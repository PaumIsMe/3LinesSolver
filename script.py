import itertools
from collections import deque
import sys
import copy
import requests
from bs4 import BeautifulSoup
import json

import cProfile
import pstats


# Debugging
print_breakdowns = False

# Route possibilities
damage_source = ['DDark', 'Shade Soul', 'VS']
initial_route = ['Lantern Route', 'Dive Route']
dv_entrance = ['DVQG', 'DVDS']
wings = ['Wings', 'Wingless']

# Global variables (should pass these every time but nahhh)
all_goals = {}
discounts = {}

# Individual Consumables
idol_consumables = {}
mask_consumables = {}
ore_consumables = {}
key_consumables = {}
essence_consumables = {}
dream_tree_consumables = {}

id_from_tax = {
    "Pale Ore": "pale_ore",
    "Essence": "essence",
    "Grubs" : "grub",
    "Vessel Fragment" : "vessel",
    "Mask Shard" : "mask",
    "Geo" : "geo",
    "Max Geo": "max_geo"
}

# Track consumables needed for tax
additional_consumables = {}

all_lines = ["R1", "R2", "R3", "R4", "R5", "C1", "C2", "C3", "C4", "C5", "TLBR", "BLTR"]

class Goal:

    def __init__(self, name, possibilities):
        self.name = name
        self.ways_to_achieve = {}
        self.current_subgoals = set()
        self.current_additional_time = 0

        # Discounts [Pale Ore][Nosk], ex
        self.discounts = {}

        # Total Consumables
        self.pale_ore = 0
        self.essence = 0
        self.grub = 0
        self.vessel = 0
        self.mask = 0
        self.geo = 0
        self.max_geo = 0

        self.set_ways_to_achieve(possibilities)

    # This happens with initialization, parses the copy-pasted tsv and makes it into something meaningful
    def set_ways_to_achieve(self, possibilities):

        for p in possibilities:

            # If this is an instance of a "special tax", then we don't want to count it as a goal, but rather extract the tax
            # We use setattr because we want to set the exact name as specified on the config input
            if "!!" in p:
                special_tax, value = (value.strip() for value in p.split("!!"))
                special_tax_id = id_from_tax[special_tax]
                value = int(value)
                setattr(self, special_tax_id, value)

            # Everything else should be a requirement
            else:

                # Extract requirements
                if "-" in p:
                    config, goals = p.split("-")
                    requirements = csv_as_set(config)
                    sub_goals = csv_as_set(goals)
                else:
                    sub_goals = csv_as_set(p)
                    requirements = set()
                
                time = 0
                for val in sub_goals:
                    if val.isdigit():
                        time = int(val)
                        sub_goals.remove(val)
                        break

                self.ways_to_achieve[tuple(requirements)] = {"sub_goals": sub_goals, "time": time}

    # Takes in a config and sets the required goals and time given that config
    def reset_subgoals(self, config):

        for requirements, goals_and_time in self.ways_to_achieve.items():
            if set(requirements).issubset(config):
                self.current_subgoals = goals_and_time["sub_goals"]
                self.current_additional_time = goals_and_time["time"]
                return 0
        
        raise KeyError(f"Goal {self.name} does not have any possibilities with config {config}")
        return -1
    
    def get_additional_goal_requirements(self, known_needed_goals):
        return self.current_subgoals - known_needed_goals
        # Goals in subgoals but not in known needed goals

    def get_name(self):
        return self.name

    def get_time(self):
        return self.current_additional_time

    def get_pale_ore(self):
        return self.pale_ore

    def get_essence(self):
        return self.essence
    
    def get_grub(self):
        return self.grub
    
    def get_vessel(self):
        return self.vessel
    
    def get_mask(self):
        return self.mask

    def get_geo(self):
        return self.geo

    def get_max_geo(self):
        return self.max_geo

    def get_discounts(self):
        return self.discounts

def values_as_set(values_string, delim):
    return {value.strip() for value in values_string.split(delim)}

def csv_as_set(csv_string):
    return values_as_set(csv_string, ",")

# Given the input of all goals and their requirements given config, put that into a comprehensive dictionary of Goals
def read_all_goals(tsv_input):

    file = open(tsv_input).read().strip() # All characters including \n
    lines = file.split("\n") # All lines

    for l in lines:

        # Ignore everything past discounts
        if "DISCOUNTS" in l:
            break

        goal_name, *requirements = l.split('\t')

        # Ignore empty lines and empty spaces
        if not goal_name:
            continue
        
        while '' in requirements:
            requirements.remove('')

        all_goals[goal_name] = Goal(goal_name, requirements)


# Given the input of all goals and their requirements given config, put that into a comprehensive dictionary of Goals
def read_discounts(tsv_input):

    file = open(tsv_input).read().strip() # All characters including \n
    lines = file.split("\n") # All lines

    discounts_section_entered = False
    global all_goals
    

    for l in lines:

        # Mark the start of discounts section
        if "DISCOUNTS" in l:
            discounts_section_entered = True
            continue

        if "CONSUMABLES" in l:
            break
        
        # Do not treat goals as discounts
        if not discounts_section_entered: 
            continue

        # Skip Empty Lines
        if not l.strip():
            continue

        discount_goals, timesave, *rest = l.split('\t')

        # Ignore empty lines and empty spaces
        if not discount_goals:
            continue
        
        discount_goals = values_as_set(discount_goals, ";")
        timesave = int(timesave)

        # Check for validity of discount goals
        for goal in discount_goals:
            if goal not in all_goals:
                print(f"Goal {goal} found in discounts but not found in all goals. Exiting...")
                exit(0)

        discounts[tuple(discount_goals)] = timesave


def read_consumables(tsv_input):

    file = open(tsv_input).read().strip() # All characters including \n
    lines = file.split("\n") # All lines

    consumables_section_entered = False

    for l in lines:

        # Mark the start of discounts section
        if "CONSUMABLES" in l:
            consumables_section_entered = True
            continue

        # Do not treat goals as discounts
        if not consumables_section_entered: 
            continue

        # Skip Empty Lines
        if not l.strip():
            continue

        consumable_name, *rest = l.split('\t')
        consumable_type, consumable_name = consumable_name.split("-")

        ways_to_obtain = {}
        for way in rest:
            if "-" in way:
                goals_needed, time_to_obtain = way.split("-")
                time_to_obtain = int(time_to_obtain.strip())
                ways_to_obtain[goals_needed.strip()] = time_to_obtain
            elif way:
                ways_to_obtain[""] = int(way)

        d = get_consumable_set_from_text(consumable_type.strip())
        d[consumable_name.strip()] = ways_to_obtain

def get_consumable_set_from_text(name):
    if name == "Mask Shard":
        return mask_consumables
    elif name == "Idol":
        return idol_consumables
    elif name == "Pale Ore":
        return ore_consumables
    elif name == "Simple Key":
        return key_consumables
    elif name == "Essence":
        return essence_consumables
    elif name == "Dream Tree":
        return dream_tree_consumables
    elif name == "Grub":
        return {}
    else:
        print(f"Unknown consumable name: {name}")
        exit()

def get_board_goals(board_file_name):
    file = open(board_file_name).read().strip()
    lines = file.split("\n")

    goal_array = [[None for i in range(5)] for j in range(5)]

    for l in lines:
        if '</div></td>' in l:
            parse_l = l[:-11]
            goal_name = parse_l.split('>')[-1]
            # print(l)
            col = int(l[l.find("col") + 3])
            row = int(l[l.find("row") + 3])
            goal_array[row - 1][col - 1] = goal_name
    
    return goal_array

def get_all_necessary_goals(goal_list):

    # All goals on the queue are alread in required goals
    to_process = deque(goal_list)

    required_goal_names = set()
    for goal in goal_list:
        required_goal_names.add(goal)

    # Go thru all goals and add any goals that we will need that we are not already aware of
    while to_process:
        goal_name = to_process.popleft()
        
        additional_goal_names = all_goals[goal_name].get_additional_goal_requirements(required_goal_names)

        for name in additional_goal_names:
            required_goal_names.add(name)
            to_process.append(name)
    
    return set(all_goals[name] for name in required_goal_names)

def get_goal_times(goal_list):

    t = 0

    for g in goal_list:
        t += g.get_time()
    return t

def tax(goal_list, method, time_per_extra):
    time, item = tax_with_count(goal_list, method, time_per_extra)
    return time

# Returns the tax and the number of the items the route will get
def tax_with_count(goal_list, method, time_per_extra):

    required_quantity = 0
    obtained_quantity = 0

    for g in goal_list:
        quantity = method(g)
        required_quantity = max(required_quantity, -1 * quantity) # Requirements are expressed as negatives, we negate to get the requirement
        
        if quantity > 0:
            obtained_quantity += quantity
    
    if (required_quantity - obtained_quantity) > 0: 
        return time_per_extra * (required_quantity - obtained_quantity), required_quantity
    else:
        return 0, obtained_quantity
    
# Returns the tax and the number of the items the route will get
def essence_tax(required_goals, method):

    required_essence = 0
    obtained_essence = 0

    for g in required_goals:
        essence = method(g)
        required_essence = max(required_essence, -1 * essence) # Requirements are expressed as negatives, we negate to get the requirement
        
        if essence > 0:
            obtained_essence += essence
    
    local_essence_consumables = copy.deepcopy(essence_consumables)
    
    # Returns (required, obtained)
    if required_essence <= obtained_essence:
        return 0, obtained_essence
    else:
        return consumable_tax_dynamic(local_essence_consumables, required_essence, required_goals), required_essence

    
# Roughly calculates the amount of geo required for the route vs the amount picked up on the way
def geo_tax(goal_list, config):

    total_obtained = 0
    total_spent = 0
    spend_requirement = 0

    for g in goal_list:
        geo_to_spend = g.get_geo()
        max_geo = g.get_max_geo()

        if geo_to_spend < 0:
            total_spent -= geo_to_spend
        
        if geo_to_spend > 0:
            total_obtained += geo_to_spend

        if max_geo < 0: # Specifically for the spend 3k goals
            spend_requirement = max(spend_requirement, -1 * max_geo)
        
        if max_geo > 0: # Specifically for the routes
            total_spent += max_geo
            total_obtained += max_geo


    final_spent = max(spend_requirement, total_spent)
    geo_debt = final_spent - total_obtained

    if geo_debt > 0:
        return (3 * geo_debt) // 100 , 0, total_obtained
    else:
        return 0, total_obtained - final_spent, total_obtained

def initialize_goals(c):
    for g in all_goals:
        all_goals[g].reset_subgoals(c)

def apply_discounts(d, goal_list):

    goal_list_names = set(g.get_name() for g in goal_list)
    for discount, time in discounts.items():
        if set(discount).issubset(goal_list_names):
            d["-".join(discount)] = -1 * time

def find_total_time(route_breakdown):
    tot = 0
    for k, v in route_breakdown.items():
        if not k[0] == "-": #Ignore non-time goals like total geo spent
            tot += v
    return tot

def get_route_breakdown(goals_on_line, config):

    goals_on_route = set(g for g in goals_on_line)

    if "Wings" in config:
        goals_on_route.add("Monarch Wings")
    if "DDark" in config:
        goals_on_route.add("Descending Dark")
    if "Shade Soul" in config:
        goals_on_route.add("Shade Soul")
    if "Lantern Route" in config:
        goals_on_route.add("Lantern Route Enjoyer")
    if "Dive Route" in config:
        goals_on_route.add("Dive Route Enjoyer")

    initialize_goals(config)
    required_goals = get_all_necessary_goals(goals_on_route)

    route_breakdown = {}

    route_breakdown["Geo Tax"], extra_geo, total_geo = geo_tax(required_goals, config) # Relics / Money
    route_breakdown["Essence Tax"], essence_obtained = essence_tax(required_goals, Goal.get_essence) # Essence
    route_breakdown["Pale Ore Tax"] = pale_ore_tax(required_goals, essence_obtained) # Pale Ore
    route_breakdown["Grub Tax"] = tax(required_goals, Goal.get_grub, 10) # Grubs
    route_breakdown["Vessel Fragment Tax"] = tax(required_goals, Goal.get_vessel, 40) # Vessel Fragments
    route_breakdown["Simple Key Tax"], extra_geo = simple_key_tax(required_goals, extra_geo)
    route_breakdown["Mask Shard Tax"], extra_geo = mask_shard_tax(required_goals, extra_geo) # Mask Shards
    route_breakdown["King's Idol Tax"] = kings_idol_tax(required_goals, essence_obtained)
    route_breakdown["Dream Tree Tax"] = dream_tree_tax(required_goals) # Dream Trees
    route_breakdown["--Total Spent Geo"] = total_geo

    determine_lemm_sell(required_goals, extra_geo)

    # Add all goals (includes base route)
    for g in required_goals:
        route_breakdown[g.name] = g.get_time()

    apply_discounts(route_breakdown, required_goals)

    return route_breakdown

def determine_lemm_sell(required_goals, extra_geo):

    if ("Dive Route Enjoyer" in required_goals and extra_geo < 1450) or ("Lantern Route Enjoyer" in required_goals and extra_geo < 800):
        required_goals.add("Lemm Sell") 

def mask_shard_tax(required_goals, extra_geo):

    acquired_goal_names = set(g.name for g in required_goals)

    if "Obtain 2 extra masks" in acquired_goal_names:
        needed_mask_shards = 8
    elif "Obtain 1 extra mask" in acquired_goal_names:
        needed_mask_shards = 4
    else:
        return 0, extra_geo

    local_mask_consumables = copy.deepcopy(mask_consumables)

    for idx, cost in enumerate([150, 500, 800, 1500]):
        if extra_geo >= cost:
            local_mask_consumables[f"Sly{idx + 1}"][""] = 8
            extra_geo -= cost
    
    return consumable_tax(local_mask_consumables, needed_mask_shards, required_goals), extra_geo

def pale_ore_tax(required_goals, essence_obtained):

    acquired_goal_names = set(g.name for g in required_goals)

    if "Nail 3" in acquired_goal_names:
        needed_pale_ore = 3
    elif "Have 2 Pale Ore" in acquired_goal_names:
        needed_pale_ore = 2
    elif "Nail 2" in acquired_goal_names:
        needed_pale_ore = 1
    else:
        return 0

    local_ore_consumables = copy.deepcopy(ore_consumables)

    if essence_obtained > 300:
        essence_obtained = 300

    local_ore_consumables["Essence"]["Dream Nail"] = 19 + (300 - essence_obtained) # TODO - fix when we have essence rewards
    local_ore_consumables["Essence"]["Seer Rewards"] = (300 - essence_obtained)

    return consumable_tax(local_ore_consumables, needed_pale_ore, required_goals)

def dream_tree_tax(required_goals):

    acquired_goal_names = set(g.name for g in required_goals)
    local_dream_tree_consumables = copy.deepcopy(dream_tree_consumables)

    if "Complete 4 full dream trees" in acquired_goal_names:
        required_dream_trees = 4
    else:
        return 0

    return consumable_tax(local_dream_tree_consumables, required_dream_trees, required_goals)

def kings_idol_tax(required_goals, essence_obtained):

    acquired_goal_names = set(g.name for g in required_goals)

    if "Collect 3 King's Idols" in acquired_goal_names:
        needed_idols = 2
    else:
        return 0

    local_idol_consumables = copy.deepcopy(idol_consumables)

    if essence_obtained < 200:
        local_idol_consumables["Glade"]["Dream Nail"] = 37 + (200 - essence_obtained) # TODO - fix when we have essence rewards
        local_idol_consumables["Glade"]["Seer Rewards"] = 18 + (200 - essence_obtained)
    
    idol_time = consumable_tax(local_idol_consumables, needed_idols, required_goals)

    return min(idol_time, 122) # If we're getting both edge idols we will never take more than this

def simple_key_tax(required_goals, extra_geo):

    acquired_goal_names = set(g.name for g in required_goals)

    needed_simple_keys = 0    

    if "WW Bench" in acquired_goal_names:
        needed_simple_keys += 1
    if "Kill your shade in Jiji's Hut" in acquired_goal_names:
        needed_simple_keys += 1
    if "Pleasure House" in acquired_goal_names:
        needed_simple_keys += 1
    if "Obtain Godtuner" in acquired_goal_names:
        needed_simple_keys += 1

    if "Use 2 Simple Keys" in acquired_goal_names:
        needed_simple_keys = max(needed_simple_keys, 2)

    if needed_simple_keys == 0:
        return 0, extra_geo
    
    local_key_consumables = copy.deepcopy(key_consumables)

    geo_savings = min(extra_geo, 950)
    extra_geo -= geo_savings
    local_key_consumables["Sly"][""] = 4 + (((950 - geo_savings) * 3) // 100)

    return consumable_tax(local_key_consumables, needed_simple_keys, required_goals), extra_geo

def has_all_goals(necessary_goals_csv, acquired_goal_names):
    if len(necessary_goals_csv) == 0:
        return True
    return csv_as_set(necessary_goals_csv).issubset(acquired_goal_names)

def consumable_tax(consumable_tax_dictioanry, extras_needed, required_goals):

    acquired_goal_names = set(g.name for g in required_goals)
    consumable_times = []
    total_time = 0

    for consumable_name, ways_to_obtain in consumable_tax_dictioanry.items():
        best_time_to_obtain = 1000

        for necessary_goals, time_for_goal in ways_to_obtain.items():
            if has_all_goals(necessary_goals, acquired_goal_names):
                best_time_to_obtain = min(best_time_to_obtain, time_for_goal)
        
        consumable_times.append( (consumable_name, best_time_to_obtain) )
    
    consumable_times.sort(key = lambda x: x[1])

    for goal_name, time in consumable_times[0:extras_needed]:
        additional_consumables[goal_name] = time
        total_time += time
        
    return total_time

# TODO - this is really just for essence, we can rename
def consumable_tax_dynamic(consumable_tax_dictioanry, extras_needed, required_goals):

    acquired_goal_names = set(g.name for g in required_goals)
    consumable_times = []

    for consumable_name, ways_to_obtain in consumable_tax_dictioanry.items():
        best_time_to_obtain = 1000

        for necessary_goals, time_for_goal in ways_to_obtain.items():
            if has_all_goals(necessary_goals, acquired_goal_names):
                best_time_to_obtain = min(best_time_to_obtain, time_for_goal)
        
        consumable_times.append( (consumable_name, best_time_to_obtain, int(consumable_name.split(" ")[-1])) )
    
    return calculate_minimum_essence_time(consumable_times, extras_needed)

# Dynamic programming to calculate min essence needed!
def calculate_minimum_essence_time(consumable_times, extras_needed):
    INF = 10**10

    # Only care up to extras_needed
    dp = [INF] * (extras_needed + 1)
    dp[0] = 0

    for _, time, reward in consumable_times:
        for r in range(extras_needed, -1, -1):
            if dp[r] == INF:
                continue
            new_r = min(extras_needed, r + reward)
            new_time = dp[r] + time
            if new_time < dp[new_r]:
                dp[new_r] = new_time

    return dp[extras_needed]


def all_possible_line_combinations(number_of_lines):
    return itertools.combinations(["R1", "R2", "R3", "R4", "R5", "C1", "C2", "C3", "C4", "C5", "TLBR", "TRBL"], number_of_lines)

def all_config_combinations():
    return itertools.product(damage_source, initial_route, dv_entrance, wings)

def update_best_times(d, lines, time, config, route_breakdown):

    if lines in d:
        if time < d[lines]["time"]:
            d[lines] = {"time": time, "config": config, "route_breakdown": route_breakdown}

    else:
        d[lines] = {"time": time, "config": config, "route_breakdown": route_breakdown}

def all_line_combinations(board_goals, num_lines):
    combinations = list(itertools.combinations(all_lines, 3))
    return combinations

def get_goal_list(lines, board_goals):

    required_goals = set()

    for line in lines:
        if line[0] == 'R':
            row_idx = int(line[1]) - 1
            for goal in board_goals[row_idx]:
                required_goals.add(goal)
        
        elif line[0] == 'C':
            col_idx = int(line[1]) - 1
            for row in board_goals:
                required_goals.add(row[col_idx])
        
        elif line == "TLBR":
            for i in range(0, 5):
                required_goals.add(board_goals[i][i])

        elif line == "BLTR":
            for i in range(0, 5):
                required_goals.add(board_goals[i][4 - i])
        
        else:
            print(f"ERROR, line {line} is not valid")
            exit(-1)
        
    return required_goals

def print_route_breakdown(route_breakdown):
    for k in sorted(route_breakdown.keys()):
        if not ("Tax" in k and route_breakdown[k] == 0): #Ignore taxes of 0
            print(f"\t\t{k} \t{route_breakdown[k]}")

def parse_user_input(user_input):
    if len(user_input) == 1:
        return None, []
    board_url = None
    lines = []
    for str_input in user_input[1:]:
        if (str_input.startswith("https://") or str_input.startswith("http://")):
            board_url = str_input
        else:
            lines.append(str_input)
            
    lines = [tuple(lines)] if lines else []
    return board_url, lines

def get_board_from_url(url):
    if url.endswith('/'):
        url = url[:-1]
    if not url.endswith('/board'):
        url += '/board'
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, 'html.parser')
        goals_json = next(iter(soup))
        parsed = json.loads(goals_json)
        goals = [square['name'] for square in parsed]
        # turn into a 5x5 matrix
        goals_matrix = [goals[i:i+5] for i in range(0, 25, 5)]
        return goals_matrix
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def print_additional_consumables():
    print("Additional required consumables:")
    for k, v in additional_consumables.items():
        print(f"{k}: {v}")

def main():

    best_times = {}

    read_all_goals("goal_requirements.tsv")
    read_discounts("goal_requirements.tsv")
    read_consumables("goal_requirements.tsv")

    board_url, user_lines = parse_user_input(sys.argv)
    if board_url is None:
        print("Loading board goals from file")
        board_goals = get_board_goals("board_goals.txt")
    else:
        print(f"Loading board goals from {board_url}")
        board_goals = get_board_from_url(board_url)
    
    print(f"Board goals detected: {board_goals}")
    
    if (user_lines):
        print(f"Using User-Submitted lines: {user_lines[0]}")
        lines_to_check = user_lines
    else:
        lines_to_check = all_line_combinations(board_goals, 3)

    for lines in lines_to_check:
        required_goals = get_goal_list(lines, board_goals)
        if print_breakdowns:
            print(required_goals)

        for config in all_config_combinations():

            additional_consumables.clear()
            route_breakdown = get_route_breakdown(required_goals, config)
            time = find_total_time(route_breakdown)

            if print_breakdowns:
                print(f"\t{config}: {time}")
                print_route_breakdown(route_breakdown)
                print_additional_consumables()

            update_best_times(best_times, lines, time, config, route_breakdown)


    best_row_combo = None
    best_route = {"time": 1000000, "config": "Invalid", "route_breakdown": "Invalid"}

    sorted_list = sorted([(key, value) for key, value in best_times.items()], key=lambda item: item[1]["time"], reverse=True)
    
    for row_combination, route in sorted_list:
        print(f"Best time for {row_combination}:\t{route['time']}")
        if route["time"] < best_route["time"]:
            best_row_combo = row_combination
            best_route = route

    print(f"Best Rows: \t {best_row_combo}")
    print(f"Estimated Time: \t {best_route['time']}")
    print(f"Suggested Route: \t {best_route['config']}")
    print(f"Time breakdown:")
    print_route_breakdown(best_route["route_breakdown"])

if __name__ == '__main__':
    #profiler = cProfile.Profile()
    #profiler.enable()
    main()
    #profiler.disable()

    #stats = pstats.Stats(profiler)
    #stats.sort_stats("cumulative")
    #stats.print_stats()