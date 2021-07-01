################################## PYTORCH DATALOADER MANAGEMENT ##################################
from torch.utils.data.dataloader import default_collate
from torch import Tensor

def collateCleanNone(batch): # for Cross-Entropy Classification...
    batch = list(filter(lambda x: x!=None, batch))
    return default_collate(batch) if len(batch)!=0 else [(Tensor(),Tensor()), Tensor()]

def collateCleanNone_floatLabel(batch): # for MSE Regression or user defined HingeLoss...
    batch = [(data, label.float()) for data, label in filter(lambda x: x!=None, batch)]
    return default_collate(batch) if len(batch)!=0 else [(Tensor(),Tensor()), Tensor()]

def collateCleanNone_flattenInput(batch): # ...over a fully-connected first layer
    batch = [(data[0].view(-1), label) for data, label in filter(lambda x: x!=None, batch)]
    return default_collate(batch) if len(batch)!=0 else [(Tensor(),Tensor()), Tensor()]

def collateCleanNone_flattenInput_floatLabel(batch): # ...over a fully-connected first layer
    batch = [(data[0].view(-1), label.float()) for data, label in filter(lambda x: x!=None, batch)]
    return default_collate(batch) if len(batch)!=0 else [(Tensor(),Tensor()), Tensor()]



########################  DATASETS CLASSES  ########################
from pandas import read_parquet, Series
from torch import is_tensor, tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from numpy import ndarray

class StoredImageBytesDataset(Dataset):
    
    def __init__(self, filePath, output_feature, sensible_feature=[], normalise_output=None, drop_sensible=None,
                 transform=ToTensor(), reduced=0, vectorise_out=False, verbose=False):
        super(StoredImageBytesDataset, self).__init__()

        self.output_feature = output_feature
        self.sensible_feature = sensible_feature
        self.transform = transform
        self.verbose = verbose
        self.full_ds = read_parquet(filePath)
        self.extracted_features = isinstance(self.full_ds.iloc[0]["image_bytes"], ndarray)
        
        
        # Cleaning -1 (undefined) values over chosen output and sensitive (if defined) attributes
        valid = (self.full_ds[output_feature]!=-1) & ((self.full_ds[self.sensible_feature]!=-1) if self.sensible_feature!=[] else True)
        self.full_ds = self.full_ds.loc[valid]
        self.sensible_pops = sorted(self.full_ds[self.sensible_feature].unique()) if self.sensible_feature!=[] else []

        # create an artificially biased dataset
        if drop_sensible:
            assert drop_sensible[1]>=0 and drop_sensible[1]<=1, "second parameter of drop_sensible must be in [0,1]"
            assert self.sensible_feature!=[], "for dropping sensible values you need to define a sensible feature first!"

            popDrop_idxs = self.full_ds.loc[self.full_ds[self.sensible_feature]==drop_sensible[0]].sample(frac=drop_sensible[1]).index
            self.full_ds = self.full_ds.drop(index=popDrop_idxs)

        # reduce dataset size
        if reduced>0 and reduced<len(self.full_ds):
            assert isinstance(reduced, int), "The reduce parameter must be a positive integer"
            self.full_ds = self.full_ds.sample(n=reduced)

        self.X = self.full_ds[["image_bytes", self.sensible_feature] if self.sensible_feature!=[] else ["image_bytes"]]
        self.y = self.full_ds[self.output_feature] 
        if vectorise_out:
            self.y = Series(data=[(1.-y,y) for y in self.y], index=self.y.index, name=self.y.name)
        
        # output normalisation
        classes = sorted(self.y.unique())
        if normalise_output=="0,1":
            minm, maxm = min(classes), max(classes)
            self.y = ((self.y-minm)/(maxm-minm)).round(decimals=2)
        elif normalise_output=="-1,1":
            minm, maxm = min(classes), max(classes)
            self.y = -1 + (2*(self.y-minm)/(maxm-minm)).round(decimals=2)

        self.output_normMapping = dict(zip(sorted(self.y.unique()), classes))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):        
        if is_tensor(index): index = index.tolist()
        data = self.X.iloc[index]["image_bytes"]
        data = tensor(data) if self.extracted_features else self.transform(Image.open(BytesIO(data)))
        return (data, tensor(self.X.iloc[index][self.sensible_feature])), tensor(self.y.iloc[index])
