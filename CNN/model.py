import torch
import torch.nn as nn
import torch.nn.functional as Func

class SudokuEncoder(nn.Module):

    def __init__(self):

        super(SudokuEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=0)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2) 

        self.conv2 = nn.Conv2d(8, 14, kernel_size=5, padding=0)
        self.bn2 = nn.BatchNorm2d(14)
        self.pool2 = nn.MaxPool2d(2, 2) 

        self.conv3 = nn.Conv2d(14, 22, kernel_size=5, padding=0)
        self.bn3 = nn.BatchNorm2d(22)
        self.pool3 = nn.MaxPool2d(2, 2) 

        self.conv4 = nn.Conv2d(22, 12, kernel_size=3, padding=0)
        self.bn4 = nn.BatchNorm2d(12)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))

    def forward(self, img_data):
        x = Func.relu(self.bn1(self.conv1(img_data)))
        x = self.pool1(x)
        x = Func.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = Func.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = Func.relu(self.bn4(self.conv4(x)))
        return self.adaptive_pool(x) 


class SudokuDecoderDigit(nn.Module):

    def __init__(self):

        super(SudokuDecoderDigit, self).__init__()
        self.flatten = nn.Flatten()
        self.MLP = nn.Linear(in_features=300, out_features=10)

    def forward(self, feature_map):

        return self.MLP(self.flatten(feature_map))


class SudokuDecoderSum(nn.Module):

    def __init__(self):

        super(SudokuDecoderSum, self).__init__()
        self.squeeze = nn.Conv2d(12, 2, kernel_size=1)
        
        self.conv_layer_row = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1,5), stride=1, padding=(0,2))
        self.conv_layer_column = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(5,1), stride=1, padding=(2,0))
        self.flatten = nn.Flatten()
        self.MLP = nn.Linear(in_features=100, out_features=6)

    def forward(self, feature_map):

        feature_map = Func.relu(self.squeeze(feature_map))
        feature_map_row = Func.relu(self.conv_layer_row(feature_map))
        feature_map_column = Func.relu(self.conv_layer_column(feature_map))

        shortcut_row = feature_map.mean(dim=3, keepdim=True)
        shortcut_col = feature_map.mean(dim=2, keepdim=True)
        feature_map_row = feature_map_row + shortcut_row
        feature_map_column = feature_map_column + shortcut_col

        flatten_row = self.flatten(feature_map_row)
        flatten_column = self.flatten(feature_map_column)
        return self.MLP(torch.cat([flatten_column, flatten_row], dim=1))


class SudokuDecoderOrder(nn.Module):

    def __init__(self):

        super(SudokuDecoderOrder, self).__init__()
        self.squeeze = nn.Conv2d(12, 2, kernel_size=1)
        
        self.conv_layer_row = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1,5), stride=1, padding=(0,2))
        self.conv_layer_column = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(5,1), stride=1, padding=(2,0))
        self.flatten = nn.Flatten()
        self.MLP = nn.Linear(in_features=100, out_features=18)

    def forward(self, feature_map):
        
        feature_map = Func.relu(self.squeeze(feature_map))
        feature_map_row = Func.relu(self.conv_layer_row(feature_map))
        feature_map_column = Func.relu(self.conv_layer_column(feature_map))

        shortcut_row = feature_map.mean(dim=3, keepdim=True)
        shortcut_col = feature_map.mean(dim=2, keepdim=True)
        feature_map_row = feature_map_row + shortcut_row
        feature_map_column = feature_map_column + shortcut_col

        flatten_row = self.flatten(feature_map_row)
        flatten_column = self.flatten(feature_map_column)
        return self.MLP(torch.cat([flatten_column, flatten_row], dim=1)).view(-1, 6, 3)


class SudokuNet(nn.Module):
    def __init__(self):
        super(SudokuNet, self).__init__()
        self.encoder = SudokuEncoder()
        self.decoder_digit_head = SudokuDecoderDigit()
        self.decoder_sum_head = SudokuDecoderSum()
        self.decoder_order_head = SudokuDecoderOrder()

    def forward(self, img_data):
        feature_map = self.encoder(img_data)
        return self.decoder_digit_head(feature_map), self.decoder_order_head(feature_map), self.decoder_sum_head(feature_map)

def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    my_net = SudokuNet()
    print(f"Total Parameters: {count_trainable_parameters(my_net)}")