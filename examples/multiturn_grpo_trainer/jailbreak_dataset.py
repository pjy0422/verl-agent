"""
Custom dataset for multi-turn jailbreak environment.
Extends RLHFDataset to add env_kwargs with behavior field.
"""
from verl.utils.dataset.rl_dataset import RLHFDataset


class JailbreakDataset(RLHFDataset):
    """
    Dataset for multi-turn jailbreak environment.

    Extends RLHFDataset to include 'env_kwargs' with the 'behavior' field
    required by multiturn_conv environment.
    """

    def __getitem__(self, item):
        """
        Get item and add env_kwargs for multiturn_conv environment.
        """
        # Get base item from parent class
        row_dict = super().__getitem__(item)

        # Extract the original dataframe row to get 'behavior' field
        df_row = self.dataframe[item]

        # Add env_kwargs required by multiturn_conv environment
        # multiturn_conv.reset() expects kwargs with 'behavior' field
        env_kwargs = {
            "behavior": df_row.get("behavior", ""),
            "data_source": df_row.get("data_source", "jailbreak"),
        }

        row_dict["env_kwargs"] = env_kwargs

        return row_dict
