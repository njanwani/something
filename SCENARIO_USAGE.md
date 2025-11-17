# Scenario-Based Motion Visualization

Generate robot motions from scenario descriptions using LLM or predefined primitives.

## Usage

### Generate New Motions with LLM

```bash
python show_scenario.py --generate --context "SCENARIO DESCRIPTION"
```

### Use Predefined Primitives (Faster)

```bash
python show_scenario.py --predefine --context "SCENARIO DESCRIPTION"
```

### From File

```bash
python show_scenario.py --generate --file scenarios/greeting_scenario.txt
python show_scenario.py --predefine --file scenarios/greeting_scenario.txt
```

### Options

- `--generate`: Generate custom motions using LLM
- `--predefine`: Use predefined motion primitives (faster, no API calls)
- `--context "text"`: Scenario description as string
- `--file path.txt`: Load scenario from file
- `--no-loop`: Play once without looping
- `--save file.py`: Save generated code to file

## Modes

### Generate Mode

Uses LLM to create custom expressive motions for each action.

- Pros: Highly expressive, custom motions
- Cons: Slower, requires API key, costs money

### Predefine Mode  

Uses pre-programmed motion primitives from `primitives/primitive.py`.

- Pros: Fast, no API costs, reliable, adjustable speed
- Cons: Limited to available primitives

Current available primitives:

- `rest` / `idle` / `stay_still`: Robot stays still
- `wave` / `wave_hello` / `wave_greeting`: Robot waves with right arm
- `frantic_wave`: Robot waves frantically with one arm
- `double_wave`: Robot waves with both arms

Note: The LLM can adjust the speed of primitives by changing the duration parameter (shorter duration = faster motion).

## Scenario Format

Describe what happens in the scene with timing. The LLM interprets this and generates appropriate robot responses.

### Example

```
A person enters the room at 0 seconds. They walk toward the robot and stop 
in front of it at 2 seconds. The person smiles and waves hello at 3 seconds. 
They chat briefly, then the person turns and walks away at 7 seconds.
```

The robot will automatically:

- Wait in idle pose initially
- Wave back when the person waves
- Return to idle when person leaves

## Included Scenarios

### greeting_scenario.txt

Person approaches, waves, and leaves.

```bash
# With LLM-generated motions
python show_scenario.py --generate --file scenarios/greeting_scenario.txt

# With predefined primitives (faster)
python show_scenario.py --predefine --file scenarios/greeting_scenario.txt
```

### pointing_scenario.txt

Person asks for directions, points to indicate location.

```bash
python show_scenario.py --generate --file scenarios/pointing_scenario.txt
```

### collaboration_scenario.txt

Person and robot work together and celebrate completion.

```bash
python show_scenario.py --generate --file scenarios/collaboration_scenario.txt
```

## Requirements

- MuJoCo and mujoco_viewer installed
- `xmls/scene.xml` model file  
- OpenAI API key in `.env` file (only for `--generate` mode)

## Adding New Primitives

Edit `primitives/primitive.py`:

1. Create a new class inheriting from `Primitive`
2. Add a CSV file with the motion trajectory
3. Add to `PRIMITIVE_REGISTRY` dictionary
4. Implement `description()` class method

Example:

```python
class Point(Primitive):
    def __init__(self, duration):
        super().__init__(
            pd.read_csv(Path(r'primitives/data/Point.csv'), index_col=0),
            duration
        )
    
    @classmethod
    def description(cls):
        return 'Robot points with right arm'

# Add to registry
PRIMITIVE_REGISTRY['point'] = Point
```

## Output

The script will:

1. Interpret the scenario using LLM
2. Plan robot actions with timing
3. Generate motion code for each action
4. Run visualization in MuJoCo
5. Loop continuously (unless --no-loop specified)

## Tips

- Include timestamps in your scenario descriptions
- Be specific about human actions and timing
- Robot automatically responds with socially appropriate behaviors
- Scenarios loop by default for repeated viewing
