import torch


class AggregationStrategy:
    def aggregate(self, clients_diff, clients_data_len):
        raise NotImplementedError()


class SimpleAverageStrategy(AggregationStrategy):
    def aggregate(self, clients_diff, clients_data_len):
        clients_num = len(clients_diff)
        weight_accumulator = {}
        for name, params in clients_diff[-1].items():
            weight_accumulator[name] = torch.zeros_like(params)

        for _, client_diff in enumerate(clients_diff):
            for name, params in client_diff.items():
                weight_accumulator[name].add_(params/clients_num)

        return weight_accumulator


class FedAverageStrategy(AggregationStrategy):
    def aggregate(self, clients_diff, clients_data_len):

        total_len = sum(clients_data_len)

        weight_accumulator = {}
        for name, params in clients_diff[0].items():
            weight_accumulator[name] = torch.zeros_like(params)

        for i, client_diff in enumerate(clients_diff):
            for name, params in client_diff.items():
                weight_accumulator[name] = weight_accumulator[name] + (params * clients_data_len[i] / total_len)

        return weight_accumulator


class MaxAverageStrategy(AggregationStrategy):
    def aggregate(self, clients_diff, clients_data_len):

        weight_accumulator = {}
        for name, params in clients_diff[-1].items():
            weight_accumulator[name] = torch.zeros_like(params)

        for _, client_diff in enumerate(clients_diff):
            for name, params in client_diff.items():
                weight_accumulator[name] = torch.where(torch.abs(params) >= torch.abs(weight_accumulator[name]),
                                                       weight_accumulator[name], params)
        return weight_accumulator
    