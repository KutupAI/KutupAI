# Data Storage Layer

Hard rule: no Agent calls any Repository directly - writes go exclusively through Application (API data) and Orchestration (state/results).
