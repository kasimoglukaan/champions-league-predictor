# champions-league-predictor
## Current Model Performance

The current model was trained on 7,592 historical football matches collected from multiple European competitions.

The evaluation was performed on 378 unseen UEFA Champions League matches using a chronological train-test split.

### Test Results

- Accuracy: 57.67%
- Log Loss: 1.0054
- Multiclass Brier Score: 0.5768

### Class Performance

| Result | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Away win | 0.49 | 0.59 | 0.53 |
| Draw | 0.45 | 0.08 | 0.14 |
| Home win | 0.64 | 0.72 | 0.68 |

The model currently performs best on home wins and has difficulty identifying draws. Planned improvements include additional form features, home/away-specific statistics, probability calibration, and high-confidence prediction evaluation.

## Data and Security

Match data is collected using the football-data.org API.

The API token is stored locally in a `.env` file and is excluded from version control. A `.env.example` file should be used to document the required environment variable:

```env
FOOTBALL_DATA_API_KEY=your_api_key_here