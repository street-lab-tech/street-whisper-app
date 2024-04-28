from backend import whisper_with_diarization_as_methods
import typer
from PyInquirer import prompt
from rich import print as rprint
from pyannote.audio import Pipeline


app = typer.Typer()

@app.command("start")
def startup_ui():
    rprint("[magenta]=============================[magenta]")
    rprint("[bold][underline]STREET Lab Whisper App[underline][bold]")
    rprint("[magenta]=============================[magenta]")
    authorization()

def authorization():
    rprint(f"What is your access token? An access token from Hugging Face and accepting pyannote/speaker-diarization-3.1's user condition is needed to run the app. If this is not your first time running the app amd have previously entered a valid token, you can just press enter on your keyboard.[bold]")
    potential_access_token = input()
    try:
        # Check token
        diarize_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=str(potential_access_token))
        questions_ui(diarize_model)
        typer.Exit()
    except KeyboardInterrupt:
        typer.Exit()
    except:
        invalidTokenPrompt = [
            {
                'type': 'list',
                'name': 'invalid_token',
                'message': 'This token is invalid. Do you want to try again?',
                'choices': [
                    {
                        'name': 'Yes',
                    },
                    {
                        'name': 'No',
                    }
                ],
            }
        ]
        invalid_token = prompt(invalidTokenPrompt)
        if invalid_token["invalid_token"] == 'Yes':
            authorization()
        else:
            typer.Exit()

def questions_ui(diarize_model):
    rprint("[magenta]=============================[magenta]")
    rprint(f"You can now go offline.[bold]")
    rprint("[magenta]=============================[magenta]")
    # Process selection
    translation_transcription_prompt = [
        {
            'type': 'list',
            'name': 'process_selected',
            'message': 'What process do you want to run on your audio file?',
            'choices': [
                {
                    'name': 'Transcription Only',
                },
                {
                    'name': 'Translation Only',
                }
            ],
        }
    ]
    process_selected = prompt(translation_transcription_prompt)
    # Input file
    rprint("[blue]=============================[blue]")
    rprint(f"[bold]Enter the absolute path to the audio file you want to do the \"{process_selected['process_selected']}\" process on:[bold]")
    input_file = input()

    rprint("[blue]=============================[blue]")
    # Is Input File in English?
    to_eng_selection_prompt = [
        {
            'type': 'list',
            'name': 'to_english_selection',
            'message': 'Is your audio file in English?',
            'choices': [
                {
                    'name': 'Yes',
                },
                {
                    'name': 'No',
                },
            ],
        }
    ]
    to_english_selection = prompt(to_eng_selection_prompt)

    rprint("[blue]=============================[blue]")
    # Model size selection
    model_size_selection_prompt = [
        {
            'type': 'list',
            'name': 'model_size_selection',
            'message': 'What model size do you want to use on your audio file?',
            'choices': [
                {
                    'name': 'large-v2',
                },
                {
                    'name': 'small',
                },
                {
                    'name': 'medium',
                },
            ],
        }
    ]
    model_size_selection = prompt(model_size_selection_prompt)
    rprint("[blue]=============================[blue]")
    # Destination Folder
    rprint(f"[bold]Enter the absolute path to your destination folder:[bold]")
    destination_selection = input()

    rprint("[blue]=============================[blue]")
    questions_finished_prompt = [
            {
                'type': 'list',
                'name': 'questions_finished',
                'message': f'Do you want to start the process? If not, this app will quit.',
                'choices': [
                            {
                                'name': 'Yes',
                            },
                            {
                                'name': 'No',
                            },
                ],
            }
        ]
    questions_finished = prompt(questions_finished_prompt)
    if questions_finished["questions_finished"] == 'Yes':
        # Run process
        whisper_with_diarization_as_methods.main(process_selected["process_selected"], input_file, to_english_selection["to_english_selection"], model_size_selection["model_size_selection"], destination_selection, diarize_model)
    else:
        # Exit out of app
        typer.Exit()

@app.command("help")
def help_ui():
    rprint("[magenta]=============================[magenta]")
    rprint("[bold][underline]STREET Lab Whisper App[underline][bold]")
    rprint("")
    rprint("[bold]Help[bold]")
    rprint("[magenta]=============================[magenta]")


if __name__ == "__main__":
    app()
