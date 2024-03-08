import sys
import fileinput
from csp import *  
from datetime import datetime

def get_operation(line , last_line) : 

    last_char = len(line) - 2        #operation is the last character ,placed before '\n'

    if last_line == line :          #last line of the file doesn't have '\n' at its end
        last_char = len(line) - 1   

    return line[last_char]

def get_number(line) :                  #line starts from its first unread character

    number = ""
    for i in range(len(line)) :

        if line[i] == '#' or line[i] =='-' :
            break
        number += str(line[i])          #Append to string as long as it's not ending character

    to_return = (int)(number)           
    return to_return

def get_variables(line) :              #line starts from its first unread character

    variables = []
    read = 0
    while read < (len(line)) :

        var = get_number(line[read :])       #get next variable from the remaining line
        read += len(str(var)) + 1            #read holds the number of the next character to be read
        variables.append(var)

        if line[read - 1] == '#' :           #if the ending character of the previous variable (hence -1) was # => no more left
            return variables   

class Grid :

    def __init__(self , operation , target , variables) :

        self.variables = variables.copy()
        self.operation = operation
        self.target = target

    
class Kenken(CSP) :

    def check (self,result,file) :

        input = open(file,'r')          #open solutions' file
        lines = input.readlines()       #get its lines
        var_num = 0
        given = {}
        for i in range(self.size) :           #for every line in the file
            variable = ""                     #holds the value of var_num variable (string with consecutive characters without space)   
            for j in range(len(lines[i])):    #for every character in that line

                if lines[i][j] == ' ' :         #we will move to next value  
                    new = int (variable)        #convert string to int
                    given[var_num] = new        #assign current string to variable var_num
                    variable = ""               #initialize string to empty to be refilled with next characters
                    var_num += 1                #next value will refer to next variable
                else:
                    variable += str(lines[i][j])        #append character to string

        if result == None :
            print("None")
            return

        check = True        
        
        for i in range(self.size * self.size) :          #check if every variable in our solution has the same value as those in file
            if(given[i] != result[i]) :
                print("Variable ",i," in given solution has value : ",given[i],", our solution: ,",result[i])
                check = False
        
        if check :
            print("EVERYTHING IS PERFECT")


    def constraint_function(self,A,a,B,b) :

        self.constraint_checks += 1
        gridA = self.var_info[A]      #holds the Grid  A corresponds to

        if A == B :                   #A is neighbor with itself if-f the operation is "="
            return a == gridA.target

        #check if they are in the same row or column 
        rowA = int (A / self.size) 
        colA = A % self.size
        rowB = int (B / self.size) 
        colB = B % self.size
        if((rowA == rowB or colA == colB) and a == b) :         #VIOLATION OF CONSTRAINT
            return False
            
        #check if they are in the same grid
        gridB = self.var_info[B]      #holds the Grid  B corresponds to
        assigned = self.infer_assignment()

        #Since they don't belong in the same grid their only mutual constraint is whether they have equal values(since they are in the same row or column)
        if gridA != gridB :
            return True       

        grid = gridA      #we will check if the combination of a with b and the already assigned variables could be acceptable

        if grid.operation == '+' :

            sum = a + b
            count = 2
            for var in  grid.variables :
                if var in assigned.keys() and var != A and var != B:
                    sum += assigned[var]
                    count += 1

            all = (sum == grid.target and count == len(grid.variables) )        #if all the variables have been assigned we should have reached the goal
            not_all = (sum < grid.target and count < len(grid.variables) )      #if we haven't assigned all the variables result should be smaller (not equal because 0 isn't a domain option for next assignments)            
            return not_all or all

        if grid.operation == '-' :     
                    
            return abs(b - a) == grid.target #subtraction has only 2 operands (based on what was said in lectures) we merely have to check if any of the combinations gives target
        
        if grid.operation == '/' :
            return (max(a,b)/min(a,b) == grid.target)           #division has only 2 operands (based on what was said in lectures) , since the result should be integer we divide bigger from smaller
    
        if grid.operation == '*' :

            mul = a * b
            count = 2         
            for var in  grid.variables :
                if var in assigned.keys() and var != A and var != B:
                    mul *= assigned[var]
                    count += 1          

            all = (mul == grid.target and count == len(grid.variables) )        #if all the variables have been assigned we should have reached the goal
            not_all = (mul <= grid.target and count < len(grid.variables) )      #if we haven't assigned all the variables result should smaller or equal (equal because in next assignments we could have 1 and it would remain acceptable)            
            return not_all or all



    """Input has the same structure as given in tests:
    size
    target # variables # operation (for every grid)"""       
    def __init__(self,file):

        input = open(file,'r')
        lines = input.readlines()
        size = (int) (lines[0]) #Get size of Kenken board
        var_to_grid = {}        #corresponds variable number to its Grid 
        variables = [i for i in range(size * size)]
        domain = [i for i in range(1,size + 1)]
        domains = {}
        neighbors = {}                      #neighbors include variables in the same row,column and grid

        for i in range(size * size) :       #initialize domain & neighbors

            domains[i] = domain
            neighbors[i] = []       

        #find all the neighbors in the same row & column
        for var in variables :
            row =  int (var / size)
            col = var % size

            #find the rest of variables in the same row
            first = row * size 
            last = first + size 
            for near in range(first , last ) :
                if near != var :
                    neighbors[var].append(near)

            #find the rest of variables in the same column
            first = col
            last = (size - 1) * size + col          
            for near in range(first, last+1 ,size) :
                if near != var :
                    neighbors[var].append(near)  

        last_line = lines[len(lines) - 1]       #last line in file

        #parse the file
        for line in lines[1:] :
            #parse the line
            target = get_number(line)                               #target is the first word
            read = len(str(target)) + 1                             #find ending character (# is estimated too in how many characters have been read, thus the +1)
            in_grid = get_variables(line[read:])                    #list of variables in grid
            operation = get_operation(line,last_line)
            new_grid = Grid(operation , target ,in_grid)

            for current in in_grid :
                for neighbor in in_grid :                        #previous & next variables in the same grid pose constraint to reach target (we only add the variables that haven't already been on the list)
                  if (current != neighbor or operation == "=") and neighbor not in neighbors[current]:
                    neighbors[current].append(neighbor)

                var_to_grid[current] = new_grid

        self.var_info = var_to_grid
        self.size = size
        self.constraint_checks = 0
        super().__init__(variables,domains,neighbors,self.constraint_function)

# Defining main function
def main():
    kenken = Kenken(sys.argv[1])
    start  = datetime.now()

    if sys.argv[3] == "FC":
        result = backtracking_search(kenken,inference=forward_checking)  		
    elif sys.argv[3] == "FC+MRV":
        result = backtracking_search(kenken, select_unassigned_variable=mrv,inference=forward_checking)   
    elif sys.argv[3] == "FC+MRV+LCV":
        result = backtracking_search(kenken, select_unassigned_variable=mrv,order_domain_values=lcv,inference=forward_checking)                 
    elif sys.argv[3] == "MIN_CON":
        result = min_conflicts(kenken)
    elif sys.argv[3] == "MAC+LCV":        
        result = backtracking_search(kenken, select_unassigned_variable=mrv,order_domain_values=lcv, inference=mac)
    elif sys.argv[3] == "MAC":        
        result = backtracking_search(kenken, inference=mac)

    finish = datetime.now()
    kenken.check(result,sys.argv[2])
    print("Algorithm took: ",finish-start," made ",kenken.nassigns," assignments ,performed ",kenken.constraint_checks," constraint checks")

if __name__=="__main__":

    main()    