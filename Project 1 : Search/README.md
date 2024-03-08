In the search.py file you will find out that all functions use three data structures ; a list keeping the explored nodes , a list for finding the path to end
goal, which we keep concatenating on every step with the new action and the
frontier for figuring out which nodes to expand which is unique for every algorithm(check implementation for more details). In most of them we also have
another list with the nodes in the frontier for easy access.The code is based
primarily on the slides’, but it doesn’t follow it precisely.
In the searchAgents.py file, I will now prove why the heuristic I have come
up with is both consistent and admissible. Notice that both 6 and 7 question
use the same one which is h(n)=maxmanhattan distance(n,y), where y is every
unvisited corner or dot respectively.
Consistency: every possible predecessor can only have moved by one in one
of the axis so since we define the cost of every function to 1: h(n)-h(n-1) <= 1
(could also take negative values, distance from furthest node could have grown
bigger) <= cost(n,n-1), since the cost has to be positive.
Admissibility: As mentioned in the implementation, we have to visit every
single corner/dot, moving only by one in either axis and therefore abs(x −
wantedx) + abs(y − wantedy) = number of actions to visit wanted state , cost
of action = min = 1 and we only get the max because in the route we might
have visited the other wanted states too, so obviously we couldn’t possibly be
overestimating. By getting the max, we keep the heuristic reliable and more
accurate.
