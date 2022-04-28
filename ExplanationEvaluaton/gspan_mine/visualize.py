import json
from ExplanationEvaluation.gspan_mine.gspan_mining.graph import Graph, Vertex, Edge
import matplotlib.pyplot as plt


class Report:
    def __init__(self, file_name):
        with open(file_name, 'r') as json_file:
            self.reports = json.load(json_file)
        for report in self.reports:
            report['dataset_name'] = report['dataset_name'].split("/")[-1].split(".")[0]
        self.reports.sort(key=lambda x: (Report._get_dataset_name(x), Report._get_pattern_id(x)))
        self.dataset_name_dict = {'aids': 'Aids', 'BBBP': 'BBBP', 'mutag': 'Mutagen'}

    @staticmethod
    def _get_dataset_name(report):
        return report['dataset_name'].split("_")[0]

    @staticmethod
    def _get_pattern_id(report):
        return int("".join(digit for digit in report['dataset_name'].split("_")[1] if digit.isdigit()))

    @staticmethod
    def _separate_vertices_edges(graph_text, mapping_function):
        graph_list = graph_text.split()
        i = 0
        edge_id = 0
        vertices = []
        edges = []
        while i < len(graph_list):
            if graph_list[i] == 'v':
                vertices.append(Vertex(vid=graph_list[i + 1], vlb=mapping_function(graph_list[i + 2])))
                i += 3
            elif graph_list[i] == 'e':
                edges.append(Edge(frm=graph_list[i + 1], to=graph_list[i + 2], elb="", eid=edge_id))
                edge_id += 1
                i += 4
        return vertices, edges

    @staticmethod
    def _build_graph(graph_text, description, mapping_function):
        vertices, edges = Report._separate_vertices_edges(graph_text, mapping_function)
        graph = Graph(gid=description)
        for vertex in vertices:
            graph.add_vertex(vertex.vid, vertex.vlb)
        for edge in edges:
            graph.add_edge(edge.eid, edge.frm, edge.to, edge.elb)
        return graph

    def visualize(self, atom_map):
        for report in self.reports:
            dataset_name = report['dataset_name'].split("_")[0]
            description = report.copy()
            del (description['graph'])
            graph = self._build_graph(graph_text=report['graph'], description="",
                                      mapping_function=lambda vertex: atom_map(dataset_name, vertex))
            print(description)
            graph.plot()

    @staticmethod
    def _draw_box_plot(numbers, labels, figure_name):
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        bp = ax.boxplot(numbers, labels=labels)
        fig.savefig(f'{figure_name}.jpg', bbox_inches='tight')

    def _group_wracc(self, extractor, x_axis):
        numbers = []
        for x in x_axis:
            numbers.append(list(map(lambda report: report['WRAcc'], filter(extractor(x), self.reports))))

        return numbers

    def draw_box_plot_for_patterns(self):
        datasets = list(set(self._get_dataset_name(report) for report in self.reports))
        numbers = self._group_wracc(extractor=lambda x: lambda report: self._get_dataset_name(report) == x,
                                    x_axis=datasets)
        self._draw_box_plot(numbers=numbers, labels=[self.dataset_name_dict[dataset] for dataset in datasets],
                            figure_name="box_plot_for_patterns")

    def draw_box_plot_for_layers(self, dataset_name, split_negative=False):
        layers, labels = ([range(20), range(20, 40), range(40, 60)], [1, 2, 3]) if not split_negative else ([
            range(i * 10, (i + 1) * 10) for i in range(6)], [(layer, decision) for layer in [1, 2, 3]
                                                             for decision in [True, False]])
        numbers = self._group_wracc(extractor=lambda x: lambda report: self._get_pattern_id(report) in x and
                                    self._get_dataset_name(report) == dataset_name,
                                    x_axis=layers)
        figure_name = f"box_plot_for_layers and {self.dataset_name_dict[dataset_name]}"
        if split_negative:
            figure_name += " with splitting decisions"
        self._draw_box_plot(numbers=numbers, labels=labels, figure_name=figure_name)
