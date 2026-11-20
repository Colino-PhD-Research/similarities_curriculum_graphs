from utility.program_structure.build_graph_from_db import build_graph
import networkx as nx
from utility.universal_helpers import get_all_universities_and_programs, get_all_universities_and_programs_by_category

university_with_cycles = []

#universities_programs = get_all_universities_and_programs()
universities_programs = get_all_universities_and_programs_by_category("Major")


for university, program in universities_programs:
    try:
        graph = build_graph(university, program, allow_messy=True)
    except Exception as e:
        print(f"{university}, {program}: error building graph- {e}")

    try:
        cycles = nx.find_cycle(graph)
        #cycles = list(nx.simple_cycles(graph))
        university_with_cycles.append((university, program, cycles))
    except:
        pass

print(f"\n\nfound {len(university_with_cycles)} programs with cycles:")
print(university_with_cycles)

