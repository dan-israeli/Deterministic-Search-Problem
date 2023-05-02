# imports
from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse import csr_matrix
import search
import json

# ids
ids = ["322794629", "209190172"]

# general constant
EMPTY, INFINITY = '', float('inf')

# constants for "assign_passengers_to_taxis" function
ASSIGNED_TAXI, DIST = 0, 1

# constants for the heuristics functions
PICKUP_COST, DROPOFF_COST, REFUEL_COST = 1, 1, 1

# constants for the "check_valid" and "result" functions
ACTION_NAME, TAXI_NAME, NEW_LOCATION, PASSENGER_NAME = 0, 1, 2, 2

# dictionaries in order to access the distance matrix
NUM_TO_COORD = {}
COORD_TO_NUM = {}


def convert_json_to_dict(json_object):
    """
    Gets a json object, turns it into a dictionary and returns it.
    """
    dictt = json.loads(json_object)
    return dictt


def convert_dict_to_json(dic):
    """
    Gets a dictionary, turns it into a json object and returns it.
    """
    json_object = json.dumps(dic)
    return json_object


class TaxiProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def check_valid(self, i, n, new_sa, sa_list, taxis):
        """
        Gets a new sub-action 'new_sa' (for example, ('pick up', 'taxi 1', 'Yossi')), and a list of other
        sub-actions, and returns whether 'new_sa' is allowed given 'sa_list'.
        """

        all_wait = True
        for sa in sa_list:
            if sa[ACTION_NAME] != "wait":
                all_wait = False

            if new_sa[ACTION_NAME] != "move" and sa[ACTION_NAME] == "move":
                if tuple(taxis[new_sa[TAXI_NAME]]["location"]) == sa[NEW_LOCATION]:
                    return False

            elif new_sa[ACTION_NAME] == "move" and sa[ACTION_NAME] != "move":
                if new_sa[NEW_LOCATION] == tuple(taxis[sa[TAXI_NAME]]["location"]):
                    return False

            elif new_sa[ACTION_NAME] == "move" and sa[ACTION_NAME] == "move":
                if new_sa[NEW_LOCATION] == sa[NEW_LOCATION]:
                    return False

        # remove the action where all taxis 'wait'
        if i == n - 1 and new_sa[ACTION_NAME] == "wait" and all_wait:
            return False

        return True

    def find_action_list(self, i, n, sa_lists, sub_res, res, taxis):
        """
        Finds all possible actions in a given state and appends them to res.
        """

        # sa = sub-action (atomic action of one taxi, out of list of actions)
        if i == n:
            sub_res = tuple(sub_res.copy())
            res.append(sub_res)
            return

        sa_list = sa_lists[i]
        for j in range(len(sa_list)):

            if self.check_valid(i, n, sa_list[j], sub_res, taxis):
                sub_res.append(sa_list[j])
                self.find_action_list(i + 1, n, sa_lists, sub_res, res, taxis)
                sub_res.pop()

    def map_coordinates_to_num(self, n, m):
        dictt = {}
        cnt = 0

        for i in range(n):
            for j in range(m):
                dictt[(i, j)] = cnt
                cnt += 1

        return dictt

    def map_num_to_coordinates(self, n, m):
        dictt = {}
        cnt = 0

        for i in range(n):
            for j in range(m):
                dictt[cnt] = (i, j)
                cnt += 1

        return dictt

    def get_distance_matrix(self, mapp):
        n, m = len(mapp), len(mapp[0])
        size = n*m

        graph = [[0 for i in range(size)] for j in range(size)]

        for i in range(size):
            (x, y) = NUM_TO_COORD[i]
            if mapp[x][y] == "I":
                continue

            # up
            if x != 0 and mapp[x - 1][y] != "I":
                j = COORD_TO_NUM[(x-1, y)]
                graph[i][j] = 1
                graph[j][i] = 1
            # down
            if x != n - 1 and mapp[x + 1][y] != "I":
                j = COORD_TO_NUM[(x+1, y)]
                graph[i][j] = 1
                graph[j][i] = 1
            # left
            if y != 0 and mapp[x][y - 1] != "I":
                j = COORD_TO_NUM[(x, y-1)]
                graph[i][j] = 1
                graph[j][i] = 1
            # right
            if y != m - 1 and mapp[x][y + 1] != "I":
                j = COORD_TO_NUM[(x, y+1)]
                graph[i][j] = 1
                graph[j][i] = 1

        graph = csr_matrix(graph)
        distance_matrix = floyd_warshall(csgraph=graph, directed=False)

        return distance_matrix.tolist()

    def get_gas_locations(self, mapp):
        gas_locations = []

        for i in range(len(mapp)):
            for j in range(len(mapp[0])):

                if mapp[i][j] == "G":
                    gas_locations.append(COORD_TO_NUM[(i, j)])

        return gas_locations

    def min_dist_with_refuel(self, gas_locations, distance_matrix, t_location, dest, fuel, is_assignment):
        min_dist = INFINITY

        for g_location in gas_locations:

            if is_assignment:
                temp_dist = distance_matrix[t_location][g_location] + distance_matrix[g_location][dest]
                min_dist = min(min_dist, temp_dist + REFUEL_COST)

            else:
                # if the taxi can reach the gas station
                if distance_matrix[t_location][g_location] <= fuel:
                    temp_dist = distance_matrix[t_location][g_location] + distance_matrix[g_location][dest]
                    min_dist = min(min_dist, temp_dist + REFUEL_COST)

        return min_dist

    def get_real_dist(self, state_dict, t_location, dest, fuel, is_assignment=False):
        gas_locations, distance_matrix = state_dict["gas_locations"], state_dict["distance_matrix"]
        dist = distance_matrix[t_location][dest]

        if fuel >= dist or (is_assignment and len(gas_locations) == 0):
            return dist

        return self.min_dist_with_refuel(gas_locations, distance_matrix, t_location, dest, fuel, is_assignment)

    def update_passengers_dict(self, initial):
        """
        :param passengers_dict:
        :return:
        picked_by_taxi - indicates which taxi picked the passenger.
        the value is None iff the passenger isn't picket yet or dropped-off in his destination
        assigned_taxi - indicates which taxi is most likely to pick up the passenger
        """
        passengers_dict = initial["passengers"]
        mapp, distance_matrix = initial["map"], initial["distance_matrix"]

        for passenger_dict in passengers_dict.values():
            passenger_dict["picked_by_taxi"] = EMPTY
            passenger_dict["assigned_taxi"] = EMPTY
            passenger_dict["assigned_taxi_dist"] = 0

            p_location = COORD_TO_NUM[tuple(passenger_dict["location"])]
            p_destination = COORD_TO_NUM[tuple(passenger_dict["destination"])]

            # there isn't a path from the passenger's location to his destination
            if distance_matrix[p_location][p_destination] == INFINITY:
                initial["solvable"] = False

    def update_taxis_dict(self, initial):
        """
        :param taxis_dict:
        :return:
        fuel_capacity - indicates the fuel capacity of the taxi
        passenger_num - indicates the number of passengers currently in the taxi
        passenger capacity - indicates the maximum number of passengers that the taxi can hold,
          it replaces the name "capacity"
        fuel - indicates the current fuel amount
        """
        taxis_dict, total_capacity = initial["taxis"], 0

        for taxi_dict in taxis_dict.values():
            taxi_dict["fuel_capacity"] = taxi_dict["fuel"]
            taxi_dict["passenger_capacity"] = taxi_dict["capacity"]

            taxi_dict["passenger_num"] = 0
            taxi_dict["assigned_passengers"] = 0

            total_capacity += taxi_dict["passenger_capacity"]

            del taxi_dict["capacity"]

        initial["total_capacity"] = total_capacity

    def update_initial(self, initial):
        """
        unpicked_passengers - indicates the total number of unpicked passengers,
        initialize to the total number of passengers in the problem
        picked_and_undelivered - indicates the total number of picked yet undelivered passengers,
        initialize to zero (no passenger is picked yet)
        """
        mapp, passengers_dict = initial["map"], initial["passengers"]

        initial["total_taxis"] = len(initial["taxis"].keys())
        initial["total_passengers"] = len(passengers_dict.keys())

        initial["unpicked_passengers"] = len(passengers_dict.keys())
        initial["picked_and_undelivered"] = 0

        initial["distance_matrix"] = self.get_distance_matrix(mapp)
        initial["gas_locations"] = self.get_gas_locations(mapp)
        initial["solvable"] = True

    def assign_passengers_to_taxis(self, state_dict):

        taxis, passengers = state_dict["taxis"], state_dict["passengers"]
        unpicked_passengers = state_dict["unpicked_passengers"]

        passengers_close_taxis = {}

        for passenger_name, passenger_dict in passengers.items():

            if passenger_dict["location"] == passenger_dict["destination"] or passenger_dict["picked_by_taxi"] != EMPTY:
                continue

            taxi_name = passenger_dict["assigned_taxi"]

            # reset assignment in order to get the best one possible in the current state
            if taxi_name != EMPTY:
                passenger_dict["assigned_taxi"] = EMPTY
                passenger_dict["assigned_taxi_dist"] = 0
                taxis[taxi_name]["assigned_passengers"] -= 1

            p_location = COORD_TO_NUM[tuple(passenger_dict["location"])]

            close_taxis = []
            for taxi_name, taxi_dict in taxis.items():
                t_location = COORD_TO_NUM[tuple(taxi_dict["location"])]

                dist = self.get_real_dist(state_dict, t_location, dest=p_location,
                                          fuel=taxi_dict["fuel"], is_assignment=True)

                close_taxis.append((taxi_name, dist))

            passengers_close_taxis[passenger_name] = sorted(close_taxis, key=lambda x: x[DIST])

        fully_assigned_taxis,  total_assigned_passengers = 0, 0

        while fully_assigned_taxis < state_dict["total_taxis"] and total_assigned_passengers < unpicked_passengers:
            fully_assigned_taxis = 0

            for taxi_name, taxi_dict in taxis.items():

                # taxi is full and can't assign anymore passengers
                if taxi_dict["assigned_passengers"] == taxi_dict["passenger_capacity"] - taxi_dict["passenger_num"]:
                    fully_assigned_taxis += 1
                    continue

                close_passengers = []
                for passenger_name, close_taxis in passengers_close_taxis.items():

                    # passenger was already assign to a taxi
                    if passengers[passenger_name]["assigned_taxi"] != EMPTY:
                        continue

                    for (taxi_name2, taxi_dist) in close_taxis:

                        # get the closest taxi to the passenger that is available to pick him up
                        if taxis[taxi_name2]["assigned_passengers"] == taxis[taxi_name2]["passenger_capacity"] - taxis[taxi_name2]["passenger_num"]:
                            continue

                        if taxi_name == taxi_name2:
                            close_passengers.append((passenger_name, taxi_dist))

                        break

                close_passengers.sort(key=lambda x: x[DIST])

                for (passenger_name, taxi_dist) in close_passengers:
                    passengers[passenger_name]["assigned_taxi"] = taxi_name
                    passengers[passenger_name]["assigned_taxi_dist"] = taxi_dist
                    taxi_dict["assigned_passengers"] += 1
                    total_assigned_passengers += 1

                    # taxi can't assign anymore passengers
                    if taxi_dict["assigned_passengers"] == taxi_dict["passenger_capacity"] - taxi_dict["passenger_num"]:
                        break

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""

        mapp = initial["map"]

        global NUM_TO_COORD, COORD_TO_NUM

        n, m = len(mapp), len(mapp[0])
        NUM_TO_COORD = self.map_num_to_coordinates(n, m)
        COORD_TO_NUM = self.map_coordinates_to_num(n, m)

        self.update_initial(initial)
        self.update_taxis_dict(initial)
        self.update_passengers_dict(initial)

        self.assign_passengers_to_taxis(initial)

        initial_json = convert_dict_to_json(initial)
        search.Problem.__init__(self, initial_json)

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""

        state_dict = convert_json_to_dict(state)

        if not state_dict["solvable"]:
            return ()

        mapp, taxis, passengers = state_dict["map"], state_dict["taxis"], state_dict["passengers"]

        # sa = sub action (an action is defined as a series of one sub-action for every taxi)
        sa_per_taxi = {}

        for taxi_name in taxis.keys():
            sa_per_taxi[taxi_name] = []

        for taxi_name, taxi_dict in taxis.items():
            (t_x, t_y) = taxi_dict["location"]

            # wait
            action_tup = ("wait", taxi_name)
            sa_per_taxi[taxi_name].append(action_tup)

            # fuel
            if mapp[t_x][t_y] == "G" and taxi_dict["fuel"] < taxi_dict["fuel_capacity"]:
                sa_per_taxi[taxi_name].append(("refuel", taxi_name))

            # pick up and drop off
            for passenger_name, passenger_dict in passengers.items():

                # indicates that the passenger arrived to their destination
                if passenger_dict["location"] == passenger_dict["destination"]:
                    continue

                # pick up
                if taxi_dict["location"] == passenger_dict["location"] and \
                        taxi_dict["passenger_num"] < taxi_dict["passenger_capacity"] and \
                        passenger_dict["picked_by_taxi"] == EMPTY:

                    sa_per_taxi[taxi_name].append(("pick up", taxi_name, passenger_name))

                # drop off
                # the taxi is at passenger's destination AND the taxi is the one that picked up the passenger
                if taxi_dict["location"] == passenger_dict["destination"] and taxi_name == passenger_dict["picked_by_taxi"]:
                    sa_per_taxi[taxi_name].append(("drop off", taxi_name, passenger_name))

            # move
            if taxi_dict["fuel"] == 0:
                continue

            n, m = len(mapp), len(mapp[0])

            # up
            if t_x != 0 and mapp[t_x-1][t_y] != "I":
                sa_per_taxi[taxi_name].append(("move", taxi_name, (t_x-1, t_y)))
            # down
            if t_x != n-1 and mapp[t_x+1][t_y] != "I":
                sa_per_taxi[taxi_name].append(("move", taxi_name, (t_x+1, t_y)))
            # left
            if t_y != 0 and mapp[t_x][t_y-1] != "I":
                sa_per_taxi[taxi_name].append(("move", taxi_name, (t_x, t_y-1)))
            # right
            if t_y != m-1 and mapp[t_x][t_y+1] != "I":
                sa_per_taxi[taxi_name].append(("move", taxi_name, (t_x, t_y+1)))

        sa_per_taxi_lists = []

        for sa_per_taxi_list in sa_per_taxi.values():
            sa_per_taxi_lists.append(sa_per_taxi_list)

        action_list = []
        self.find_action_list(0, len(sa_per_taxi_lists), sa_per_taxi_lists, [], action_list, taxis)

        return tuple(action_list)

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""

        new_state_dict = convert_json_to_dict(state)

        taxis, passengers = new_state_dict["taxis"], new_state_dict["passengers"]

        # sa = sub action (an action is defined as a series of one sub-action for every taxi)
        for sa in action:
            sa_name, taxi_name = sa[ACTION_NAME], sa[TAXI_NAME]

            if sa_name == "wait":
                continue

            elif sa_name == "refuel":
                taxis[taxi_name]["fuel"] = taxis[taxi_name]["fuel_capacity"]

            elif sa_name == "pick up":
                passenger_name = sa[PASSENGER_NAME]

                passengers[passenger_name]["picked_by_taxi"] = taxi_name
                taxis[taxi_name]["passenger_num"] += 1
                new_state_dict["picked_and_undelivered"] += 1
                new_state_dict["unpicked_passengers"] -= 1

                assigned_taxi_name = passengers[passenger_name]["assigned_taxi"]

                if assigned_taxi_name != EMPTY:
                    taxis[assigned_taxi_name]["assigned_passengers"] -= 1
                    passengers[passenger_name]["assigned_taxi"] = EMPTY
                    passengers[passenger_name]["assigned_taxi_dist"] = 0

            elif sa_name == "drop off":
                passenger_name = sa[PASSENGER_NAME]

                passengers[passenger_name]["location"] = passengers[passenger_name]["destination"]
                passengers[passenger_name]["picked_by_taxi"] = EMPTY
                taxis[taxi_name]["passenger_num"] -= 1
                new_state_dict["picked_and_undelivered"] -= 1

            elif sa_name == "move":
                taxis[taxi_name]["location"] = sa[NEW_LOCATION]
                taxis[taxi_name]["fuel"] -= 1

        self.assign_passengers_to_taxis(new_state_dict)

        new_state = convert_dict_to_json(new_state_dict)
        return new_state

    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""

        state_dict = convert_json_to_dict(state)
        unpicked_passengers, picked_and_undelivered = state_dict["unpicked_passengers"], state_dict["picked_and_undelivered"]

        return unpicked_passengers == 0 and picked_and_undelivered == 0

    def new_h(self, node):
        state_dict = convert_json_to_dict(node.state)

        mapp, taxis, passengers = state_dict["map"], state_dict["taxis"], state_dict["passengers"]
        distance_matrix = state_dict["distance_matrix"]

        total_dist = 0
        for passenger_dict in passengers.values():

            if passenger_dict["location"] == passenger_dict["destination"]:
                continue

            p_destination = COORD_TO_NUM[tuple(passenger_dict["destination"])]

            if passenger_dict["picked_by_taxi"] == EMPTY:
                p_location = COORD_TO_NUM[tuple(passenger_dict["location"])]
                dist_from_dest = distance_matrix[p_location][p_destination]

                dist_passenger_taxi = 0
                if passenger_dict["assigned_taxi"] != EMPTY and passenger_dict["assigned_taxi_dist"] != INFINITY:

                    dist_passenger_taxi = passenger_dict["assigned_taxi_dist"]

                total_dist += dist_passenger_taxi + dist_from_dest + PICKUP_COST + DROPOFF_COST

            else:
                taxi_name = passenger_dict["picked_by_taxi"]
                taxi_dict = taxis[taxi_name]
                (t_x, t_y) = taxi_dict["location"]

                # if the taxi is out of fuel, it can't refuel, and it picked a passenger that his destination is not
                # the taxi's current location, then the taxi wouldn't be able to drop off the passenger.
                # therefore, this state is not goal achievable.
                if taxi_dict["fuel"] == 0 and mapp[t_x][t_y] != "G" and taxi_dict["location"] != passenger_dict["destination"]:
                    return INFINITY

                t_location = COORD_TO_NUM[tuple(taxis[taxi_name]["location"])]
                total_dist += self.get_real_dist(state_dict, t_location,
                                                 dest=p_destination, fuel=taxi_dict["fuel"]) + DROPOFF_COST

        return total_dist / state_dict["total_taxis"]

    def new_h_2(self, node):
        """
        This is a slightly more sophisticated Manhattan heuristic
        """
        state_dict = convert_json_to_dict(node.state)

        taxis, passengers = state_dict["taxis"], state_dict["passengers"]
        distance_matrix = state_dict["distance_matrix"]

        total_dist = 0
        for passenger_dict in passengers.values():

            # passenger arrive to his destination
            if passenger_dict["location"] == passenger_dict["destination"]:
                continue

            p_destination = COORD_TO_NUM[tuple(passenger_dict["destination"])]

            # was never picked by taxi (unpicked)
            if passenger_dict["picked_by_taxi"] == EMPTY:
                p_location = COORD_TO_NUM[tuple(passenger_dict["location"])]
                total_dist += distance_matrix[p_location][p_destination] + PICKUP_COST + DROPOFF_COST

            # currently in a taxi (picked and undelivered)
            else:
                taxi_name = passenger_dict["picked_by_taxi"]
                t_location = COORD_TO_NUM[tuple(taxis[taxi_name]["location"])]
                total_dist += distance_matrix[t_location][p_destination] + DROPOFF_COST

        min_divider = min(state_dict["total_passengers"], state_dict["total_capacity"])
        return total_dist / min_divider

    def h(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state) and returns a goal distance estimate
        """
        state_dict = convert_json_to_dict(node.state)

        if state_dict["total_taxis"] == 1:
            return self.new_h_2(node)

        return self.new_h(node)

    def h_1(self, node):
        """
        This is a simple heuristic
        """
        state_dict = convert_json_to_dict(node.state)

        return (state_dict["unpicked_passengers"]*2 + state_dict["picked_and_undelivered"]) / state_dict["total_taxis"]

    def h_2(self, node):
        """
        This is a slightly more sophisticated Manhattan heuristic
        """
        state_dict = convert_json_to_dict(node.state)

        passengers, taxis = state_dict["passengers"], state_dict["taxis"]
        unpicked_dist, picked_undelivered_dist = 0, 0

        for passenger_dict in passengers.values():

            # passenger arrive to his destination
            if passenger_dict["location"] == passenger_dict["destination"]:
                continue

            (d_x, d_y) = passenger_dict["destination"]

            # was never picked by a taxi (unpicked)
            if passenger_dict["picked_by_taxi"] == EMPTY:
                (p_x, p_y) = passenger_dict["location"]
                unpicked_dist += abs(p_x - d_x) + abs(p_y - d_y)

            # currently in a taxi (picked and undelivered)
            else:
                taxi_name = passenger_dict["picked_by_taxi"]
                (t_x, t_y) = taxis[taxi_name]["location"]
                picked_undelivered_dist += abs(t_x - d_x) + abs(t_y - d_y)

        return (unpicked_dist + picked_undelivered_dist) / state_dict["total_taxis"]

    """Feel free to add your own functions
    (-2, -2, None) means there was a timeout"""


def create_taxi_problem(game):
    return TaxiProblem(game)
