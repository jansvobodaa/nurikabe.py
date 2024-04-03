import sys
# obecne
def findNeighbors(matrix,x,y):
    directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    neighbors = []
    for direction in directions:
        dx, dy = x + direction[0], y + direction[1]
        if 0 <= dx < len(matrix) and 0 <= dy < len(matrix[0]):
            neighbors.append([dx, dy])
    return neighbors

def findDiagonalNeighbors(matrix,x,y):              #hleda diagonalni sousedy
    directions = [[-1, -1], [+1, -1], [-1, +1], [+1, +1]]
    neighbors = []
    for direction in directions:
        dx,dy = x+direction[0],y+direction[1]
        if 0 <= dx < len(matrix) and 0 <= dy < len(matrix[0]) and matrix[dx][dy] > 1:
            neighbors.append([dx,dy])
    return neighbors

def findCommonNeighbors(matrix, x, y, nx, ny):          #hleda spolecne sousedy 2 bunek
    xy_neighbors = findNeighbors(matrix, x, y)
    nxy_neighbors = findNeighbors(matrix, nx, ny)
    common = []
    for i in range(len(xy_neighbors)):
        for l in range(len(nxy_neighbors)):
            if xy_neighbors[i] == nxy_neighbors[l]:
                common.append(xy_neighbors[i])
    return common

def oneIslandNeighbors(board, x, y):    #zjistuje jestli ma bunka za souseda pouze 1 ostrovni bunku
    count = 0
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        if 0 <= x + dx < len(board) and 0 <= y + dy < len(board[0]) and board[x + dx][y + dy] > 0:
            count += 1
            if count > 1:
                return False
    return True

def onlyOneIslandNeighbor(matrix,i,l):
    if matrix[i][l] == -1:
        oneNeighborIslands = []
        neighbors = findNeighbors(matrix,i,l)
        for neighbor in neighbors:
            ni, nl = neighbor
            if matrix[ni][nl] > 0:
                oneNeighborIslands.append([ni,nl])
        if len(oneNeighborIslands)> 1:
            return False
        else:
            return True

def calculateIslandSize(matrix, i, j): # dfs na zjisteni delky ostrovu
    visited = set()
    def dfsIsland(matrix, x, y, visited):
        if (x, y) in visited or matrix[x][y] <= 0:
            return 0
        visited.add((x, y))
        size = 1  #zapocitani aktualni bunky na ktere se spousti dfs
        for dx, dy in findNeighbors(matrix, x, y):
            if (dx, dy) not in visited:
                size += dfsIsland(matrix, dx, dy, visited)  # zvetsovani delky ostrovu
        return size
    return dfsIsland(matrix, i, j, visited)

def printMatrix(matrix):
    print('\n'.join([' '.join([str(i) for i in row]) for row in matrix ]))


# valid______________________________________________________________________________________________________________
def isConnected(matrix,value):               #dfs na zjisteni spojitosti pozadovane hodnoty
    def dfs(matrix, x, y, visited):
        if (x, y) in visited or matrix[x][y] != value:
            return
        visited.add((x, y))
        for dx, dy in findNeighbors(matrix, x, y):
            dfs(matrix, dx, dy, visited)
    if not matrix or not matrix[0]:
        return True
    visited = set()
    start_found = False
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == value:
                if not start_found:
                    dfs(matrix, i, j, visited)
                    start_found = True
                elif (i, j) not in visited:
                    return False  #vraci false jelikoz nalezne nespojenou bunku
    return True #hodnoty jsou spojite

def dontFormSquareWater(matrix):
    rows = len(matrix)
    cols = len(matrix[0]) if matrix else 0
    for x in range(rows - 1):
        for y in range(cols - 1): 
            if matrix[x][y] == 0 and matrix[x+1][y] == 0 and matrix[x][y+1] == 0 and matrix[x+1][y+1] == 0:
                return False
    return True

def allFilled(matrix):
    rows, cols = len(matrix), len(matrix[0])
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] < 0:
                return False
    return True

def isValidSolution(matrix):
    if dontFormSquareWater(matrix) and isConnected(matrix,0) and are_all_islands_complete(matrix) and are_islands_separated(matrix):
        return True
    return False

#delka ostrovu:___________________________________________________________
def calculate_island_size(matrix, i, j):
        visited = set()
        def dfs(x, y):
            if (x, y) in visited or matrix[x][y] <= 0:
                return 0
            visited.add((x, y))
            size = 1
            for nx, ny in findNeighbors(matrix, x, y):
                if matrix[nx][ny] == matrix[i][j]:
                    size += dfs(nx, ny)
            return size,visited
        return dfs(i, j)


def are_all_islands_complete(matrix):
    rows, cols = len(matrix), len(matrix[0])
    visited = set()
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] > 1:
                visited.add((i,j))
                if calculateIslandSize(matrix, i, j) != matrix[i][j]:
                    return False
    return True

def are_islands_separated(matrix):
    rows, cols = len(matrix), len(matrix[0])
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] >= 1:
                island_num = matrix[i][j]
                for nx, ny in findNeighbors(matrix, i, j):
                    if matrix[nx][ny] > 1 and matrix[nx][ny] != island_num or calculateIslandSize(matrix,i,j) > matrix[i][j]:
                        return False
    return True
#_________________________________________________
def onesSolution(matrix):      # pravidlo "1. Island of 1"  (https://www.conceptispuzzles.com/index.aspx?uri=puzzle/nurikabe/techniques)
    for i in range(len(matrix)):
        for l in range(len(matrix[0])):
            if matrix[i][l] == 1:
                neighbors = findNeighbors(matrix, i, l)
                for nb in neighbors:
                    matrix[nb[0]][nb[1]] = 0
    return matrix

def diagonaLSolution(matrix):             # resi diagonalni pravidlo "Neighbor Numbers"    (https://www.logicgamesonline.com/nurikabe/tutorial.html)
    rows, cols = len(matrix), len(matrix[0])
    for x in range(rows):
        for y in range(cols):
            if matrix[x][y] > 0:  #ostrov
                diagonal_neighbors = findDiagonalNeighbors(matrix, x, y)
                for nx, ny in diagonal_neighbors:
                    if matrix[nx][ny] > 0:  # diagonalne sousedni ostrov
                        common = findCommonNeighbors(matrix, x, y, nx, ny)
                        for cx, cy in common:
                            if matrix[cx][cy] == -1:  #oznaceni prazdne bunky jako voda
                                matrix[cx][cy] = 0
    return matrix

def inbetweenSolution(matrix):    #pravidlo "2. Clues separated by one square" (https://www.conceptispuzzles.com/index.aspx?uri=puzzle/nurikabe/techniques)
    rows, cols = len(matrix), len(matrix[0])
    for i in range(rows):
        for l in range(cols):
            neighbors = findNeighbors(matrix, i, l)
            numberedNeighbors = [matrix[n[0]][n[1]] for n in neighbors if matrix[n[0]][n[1]] > 0 and matrix[i][l] < 0]
            if len(numberedNeighbors) >= 2:
                matrix[i][l] = 0 
    return matrix

def Expansion(matrix):      # pravidlo Single Expansion Route (https://www.logicgamesonline.com/nurikabe/tutorial.html)
    rows, cols = len(matrix), len(matrix[0])

    changes_made = True
    while changes_made:
        changes_made = False
        for i in range(rows):
            for l in range(cols):
                if matrix[i][l] > 1:  #bunka je soucasti ostrovu ktery muze potencionalne expandovat
                    if expandIslandOneRoute(matrix, i, l):
                        changes_made = True
    return matrix

def expandIslandOneRoute(matrix, x, y): # pomocna funkce oneRouteExpansion
    island_num = matrix[x][y]
    neighbors = findNeighbors(matrix, x, y)
    negative_one_neighbors = [ng for ng in neighbors if matrix[ng[0]][ng[1]] == -1]

    for nx, ny in negative_one_neighbors:
        if canExpandTo(matrix, nx, ny, island_num):
            matrix[nx][ny] = island_num
            return True
    return False

def canExpandTo(matrix, x, y, island_num):        #pomocna funkce oneRouteExpansion
    if not isAdjacentToOtherIsland(matrix, x, y, island_num):
        original_value = matrix[x][y]
        matrix[x][y] = island_num
        if calculateIslandSize(matrix, x, y) <= island_num and are_water_and_empty_connected(matrix):
            matrix[x][y] = original_value
            return True
        matrix[x][y] = original_value
    return False

def isAdjacentToOtherIsland(matrix, x, y, island_num):         #pomocna funkce oneRouteExpansion
    for nx, ny in findNeighbors(matrix, x, y):
        if matrix[nx][ny] == 0 or (matrix[nx][ny] > 1 and matrix[nx][ny] != island_num):
            return True
    return False

def are_water_and_empty_connected(matrix):      # pomocna podminka pro doplnovani bunky pokud uzavru ostrovem -1 nebo 0
    rows, cols = len(matrix), len(matrix[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    # najit pocatecni bod pro dfs
    start_found = False
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 0 or matrix[i][j] == -1:
                start_x, start_y = i, j
                start_found = True
                break
        if start_found:
            break
    # pokud neni nalezen zadny pocatecni bod neexistuje zadna bunka s hodnotou 0 nebo -1
    if not start_found:
        return True
    def dfs(x, y):
        if not (0 <= x < rows and 0 <= y < cols) or (matrix[x][y] != 0 and matrix[x][y] != -1) or visited[x][y]:
            return
        visited[x][y] = True
        # prozkoumat sousedni bunky
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            dfs(nx, ny)
    # spustit dfs
    dfs(start_x, start_y)
    # kontrola jestli byly vsechny bunky s hodnotou 0 nebo -1 navstiveny
    for i in range(rows):
        for j in range(cols):
            if (matrix[i][j] == 0 or matrix[i][j] == -1) and not visited[i][j]:
                return False
    return True
#_________________________________________

def isolatedWater(matrix): 
    rows, cols = len(matrix), len(matrix[0])
    for i in range(rows):
        for l in range(cols):
            if matrix[i][l] == -1:
                neighbors = findNeighbors(matrix, i, l)
                if all(matrix[nx][ny] == 0 for nx, ny in neighbors):
                    matrix[i][l] = 0

def surroundIsland(matrix):     #pravidlo   "8. Surrounding a completed island"  (https://www.conceptispuzzles.com/index.aspx?uri=puzzle/nurikabe/techniques)
    def surroundWithWall(matrix, x, y, visited):
        for dx, dy in findNeighbors(matrix, x, y):
            if matrix[dx][dy] == -1 and (dx, dy) not in visited:
                matrix[dx][dy] = 0 
    rows, cols = len(matrix), len(matrix[0])
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] > 0:
                size = calculateIslandSize(matrix, i, j)
                if size == matrix[i][j]:
                    visited = set()
                    surroundWithWall(matrix, i, j, visited)
    return matrix

def emptycellsfix(matrix):
    rows, cols = len(matrix), len(matrix[0])
    for i in range(rows):
        for j in range(cols):
            neighbors = findNeighbors(matrix, i, j)
            if matrix[i][j] == -1:
                for nx, ny in neighbors:
                    if matrix[nx][ny] > 1 and calculateIslandSize(matrix,nx,ny) < matrix[nx][ny]:
                        matrix[i][j] = matrix[nx][ny]
                        if not are_water_and_empty_connected(matrix):
                            matrix[i][j] = 0

#pravidla____________________
def clues(matrix):
    onesSolution(matrix)
    inbetweenSolution(matrix)
    diagonaLSolution(matrix)
    surroundIsland(matrix)
    isolatedWater(matrix)
#_______________________________


def find_2x2_blocks(matrix):
    rows, cols =len(matrix), len(matrix[0])
    blocks = [] 

    for i in range(rows - 1):
        for j in range(cols - 1):
            if matrix[i][j] == 0 and matrix[i][j+1] == 0 and matrix[i+1][j] == 0 and matrix[i+1][j+1] == 0:
                block_cells = [(i, j), (i, j+1), (i+1, j), (i+1, j+1)]
                blocks.append(block_cells)
    return blocks

def find_neighbors_with_same_value(matrix, i, j):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] 
    visited = set() 
    same_value_neighbors = [] 
    def dfs(x, y):
        if x < 0 or x >= len(matrix) or y < 0 or y >= len(matrix[0]) or (x, y) in visited:
            return
        visited.add((x, y)) 
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(matrix) and 0 <= ny < len(matrix[0]) and matrix[nx][ny] == matrix[i][j]:
                if (nx, ny) not in visited: 
                    same_value_neighbors.append((nx, ny))
                    dfs(nx, ny)
    dfs(i, j)
    return same_value_neighbors

def find_islands(matrix):
    if not matrix or not matrix[0]:
        return []

    num_rows, num_cols = len(matrix), len(matrix[0])
    visited = set()
    islands = []
    def dfs(x, y, island):
        if (x, y) in visited or matrix[x][y] == 0:
            return
        visited.add((x, y))
        island.append((x, y))
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < num_rows and 0 <= ny < num_cols:
                dfs(nx, ny, island)
    for i in range(num_rows):
        for j in range(num_cols):
            if matrix[i][j] != 0 and (i, j) not in visited:
                island = []
                dfs(i, j, island)
                islands.append(island)
    return islands

def findSarterpoints(matrix):
    islandM = set()
    rows, cols = len(matrix), len(matrix[0])
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] >= 1:
                islandM.add((i,j))
    islandM.add(0)
    return list(islandM)

def find_moveable_cells(islands, starting_cells):
    moveable_cells = []
    def dfs(remaining_island, start, visited):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        stack = [start]
        while stack:
            cell = stack.pop()
            if cell not in visited:
                visited.add(cell)
                for dx, dy in directions:
                    next_cell = (cell[0] + dx, cell[1] + dy)
                    if next_cell in remaining_island and next_cell not in visited:
                        stack.append(next_cell)
    for island in islands:
        for cell in island:
            if cell in starting_cells:
                continue
            remaining_island = set(island) - {cell}
            if not remaining_island:
                continue
            visited = set()
            dfs(remaining_island, next(iter(remaining_island)), visited)
            if len(visited) == len(remaining_island):
                moveable_cells.append(cell)
    return moveable_cells

def place_in_water_block(grid, moveable_cells):
    def find_water_blocks(grid):
        water_blocks = []
        for i in range(len(grid) - 1):
            for j in range(len(grid[0]) - 1):
                if grid[i][j] == 0 and grid[i+1][j] == 0 and grid[i][j+1] == 0 and grid[i+1][j+1] == 0:
                    water_blocks.append(((i, j), (i+1, j), (i, j+1), (i+1, j+1)))
        return water_blocks

    def is_adjacent_to_own_island(islands, cell, new_position):
        for island in islands:
            if cell in island:
                for x, y in island:
                    if abs(new_position[0] - x) <= 1 and abs(new_position[1] - y) <= 1:
                        return True
        return False

    islands = find_islands(grid)
    water_blocks = find_water_blocks(grid)
    placements = []

    for cell in moveable_cells:
        cell_value = grid[cell[0]][cell[1]]
        for block in water_blocks:
            for position in block:
                if position in block and grid[position[0]][position[1]] == 0:
                    if is_adjacent_to_own_island(islands, cell, position):
                        new_grid = [row[:] for row in grid]
                        new_grid[cell[0]][cell[1]] = 0
                        new_grid[position[0]][position[1]] = cell_value

                        placements.append(new_grid)
    return placements


def isValid(matrix):
    if isConnected(matrix,0):
        if are_all_islands_complete(matrix):
            if are_islands_separated(matrix):
                return True
    return False

matrix = [[-1,2,-1],[-1,-1,-1],[-1,-1,-1]]
matrix = [[-1,-1,-1],[3,-1,3],[-1,-1,-1]]
xmatrix = [
    [-1, -1, -1, -1, 2, -1, 2, -1],
    [5, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [1, -1, -1, -1, 3, -1, 2, -1],
    [-1, 3, -1, -1, -1, 1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, 3],
    [-1, -1, -1, -1, -1, 4, -1, -1]
]
xmatrix = [
    [-1, 2, -1, -1, -1],
    [-1, -1, -1, 1, -1],
    [-1, -1, -1, -1, -1],
    [-1, 2, -1, -1, -1],
    [-1, -1, -1, 6, -1]
]
xmatrix = [[-1, -1, -1, 1],
[-1, -1, -1, -1],
[-1, -1, 2, -1],
[4, -1, -1, -1]]

#matrix = sys.argv[1]

staring_cells = findSarterpoints(matrix)
clues(matrix)
Expansion(matrix)
surroundIsland(matrix)
isolatedWater(matrix)
emptycellsfix(matrix)
blocks = find_2x2_blocks(matrix)
printMatrix(matrix)
def findSol(matrix):
    if len(blocks) > 0:
        acti = len(blocks)
        islands = find_islands(matrix)
        move = find_moveable_cells(islands,staring_cells)
        moves = place_in_water_block(matrix, move)
        for i in range(len(moves)):
            blockcount = find_2x2_blocks(moves[i])
            if len(blockcount)<acti and are_all_islands_complete(moves[i]): 
                print("__________________")
                printMatrix(moves[i])
            if isValid(moves[i]) and dontFormSquareWater(moves[i]):
                print("________valid__________")
                printMatrix(moves[i])
                print("________valid__________")
    else:
        pass#printMatrix(matrix)
findSol(matrix)