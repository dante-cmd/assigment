# Assignment Scheduler using Monte Carlo Tree Search (MCTS)

This project implements an automated assignment scheduling system using Monte Carlo Tree Search (MCTS) to optimize classroom allocation for educational courses. The system assigns courses to classrooms considering capacity, scheduling conflicts, and other practical constraints.

## Overview

The application solves the classroom assignment problem by:
- Allocating courses to appropriate classrooms based on capacity requirements
- Avoiding scheduling conflicts by checking time slot availability
- Optimizing the allocation process through MCTS algorithms

## Project Structure

```
assigment/
├── main.py                    # Sample Python script (not used in core functionality)
├── mcts_assignments.py        # Main MCTS algorithm implementation
├── project/                   # Data directory
│   ├── dim_aulas.json         # Classroom data with capacity (aforo)
│   ├── dim_frecuencia.json    # Frequency definitions (diario, sabatino, etc.)
│   ├── dim_horario.json       # Time slot definitions
│   ├── dim_periodo_franja.json # Time period and day definitions
│   ├── items.json             # Course data to be assigned
│   └── items_bimestral.json   # Already assigned courses (baseline schedule)
└── assignments_{location}.xlsx # Output file with assignment results
```

## Data Description

- **dim_aulas.json**: Contains classroom information for each location with their names and capacities (aforo)
- **dim_frecuencia.json**: Defines scheduling frequencies including daily (Diario), weekend (Sabatino), and other patterns
- **dim_horario.json**: Maps course time ranges to individual time slots
- **dim_periodo_franja.json**: Defines time periods and corresponding time slots for different days
- **items.json**: Contains data for courses that need to be assigned to classrooms
- **items_bimestral.json**: Contains data for already assigned courses (used to avoid conflicts)

## How It Works

The MCTS algorithm works in the following way:

1. **Initialization**: Creates a dataset with all necessary information and builds a RoomLog to track classroom availability
2. **MCTS Process**: Uses four phases:
   - Selection: Chooses the next node based on UCT (Upper Confidence bounds for Trees) algorithm
   - Expansion: Adds new nodes for unexplored actions
   - Simulation: Simulates random plays from the selected node
   - Backpropagation: Updates the node statistics based on simulation results
3. **Parallel Processing**: Uses multiple workers to run MCTS in parallel for better performance
4. **Assignment**: For each course, selects the best classroom assignment based on the MCTS results

## Key Classes

- **DataSet**: Loads and manages all data files
- **RoomLog**: Tracks classroom availability and simulates assignment actions
- **Node**: Represents nodes in the MCTS search tree
- **UCT**: Implements the core MCTS algorithm

## Usage

Run the project using:

```bash
python mcts_assignments.py --sede "Chimbote"
```

The output will be saved to `assignments_{location}.xlsx`  with the assignment results for each course.

## Reward System

The algorithm assigns rewards based on:
- If classroom capacity is less than required students: `aforo - alumnos - 2`
- If capacity is sufficient but tight (within 2 of student count): `1 + (students / capacity)`
- Otherwise: `0`

## Locations

The dataset includes classroom assignments for multiple locations:
- Chimbote
- Chincha
- Ica
- Iquitos
- La Molina
- Lima Centro
- Lima Norte Satélite
- Lima Norte Satélite 2
- Miraflores
- Pucallpa
- San Juan de Lurigancho Satélite
- San Juan de Miraflores Satélite
- San Miguel
- Surco
- ...

## Parallel Processing

The implementation includes a parallel version of MCTS that distributes the work across multiple workers to improve performance. The `parallel_UCT` function utilizes multiple processes to explore the search space concurrently.

## Output

The final assignment results are exported to an Excel file (`assignments_{location}.xlsx`) containing:
- Course information
- Assigned classroom
- Classroom capacity
- Other relevant scheduling details