# March Madness 2021                                                           
Using a [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) and [Gradient Boosting Regressor](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html) to predict NCAA March Madness game outcomes  

## Usage

There are two main scripts, one that uses Kaggle data and one that uses sports.py data. To execute these scripts, run them in the terminal as follows:

```bash
python main_march_madness.py 'Michigan' 'UCLA'
```
If you are using [main_march_madness_sportspy.py](https://github.com/bszek213/MarchMadness2021/blob/main/main_march_madness_sportspy.py) then refer to the [teamnames.txt](https://github.com/bszek213/MarchMadness2021/blob/main/teamnames.txt) txt file on the correct format for the input team name arguments (use the names within parenthesis). If you are using  [main_march_madness_kaggle.py](https://github.com/bszek213/MarchMadness2021/blob/main/main_march_madness_kaggle.py) then the standard nomenclature will suffice for your input team name arguments.
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

