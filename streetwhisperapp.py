from backend import whisper_with_diarization_as_methods
import os
import sys
import typer
from PyInquirer import prompt
from rich import print as rprint
from pyannote.audio import Pipeline


app = typer.Typer()

def startup_ui(howtouse: bool = typer.Option(False, '-howtouse', help="How to use the tool"),
               credits: bool = typer.Option(False, '-credits', help="Credits")):
    if not howtouse and not credits:
        # When no option is passed in, the app will start
        rprint("[magenta]=============================[magenta]")
        rprint("[bold][underline]STREET Lab Whisper App[underline][bold]")
        rprint("[magenta]=============================[magenta]")
        authorization()
    if howtouse and not credits:
        # When -howtouse is used, it will display the help section
        howtouse_ui()
    if not howtouse and credits:
        # When -credits is used, it will display the credits section
        credits_ui()

def validate_path(input_path: str, is_intended_file: bool) -> bool:
    """
    This function checks whether the path specified by string: input_path
    is defined in the user's system.

    If is_intended_audio_file is true, then the expected path should point to a file.
    We can adjust the testing of the file_path based on this parameter

    TBD: It also checks whether the user has the valid permissions corresponding to the path
    """
    if (is_intended_file):
        # In this branch, check whether path points to a valid file
        # We test that this file is an audio file later on (in validate_audio_file)
        is_file = os.path.isfile(input_path)
        if is_file:
            return True
        else:
            return False
    else:
        # In this branch, check whether path points to a valid directory
        is_directory = os.path.isdir(input_path)
        if is_directory:
            return True
        else:
            return False
def authorization():
    access_token_prompt = [
        {
            'type': 'password',
            'message': 'What is your access token? An access token from Hugging Face and accepting \npyannote/speaker-diarization-3.1\'s user condition is needed to run the app.\nIf this is not your first time running the app and you have previously entered a\nvalid token, press enter on your keyboard.\n',
            'name': 'password'
        }
    ]
    potential_access_token = prompt(access_token_prompt)
    if (potential_access_token == {} or potential_access_token["password"].lower() == "exit"):
        return
    try:
        # Check token
        diarize_model = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=str(potential_access_token["password"]))
        questions_ui(diarize_model)
        typer.Exit()
    except KeyError:
        # You reach here if you click on a selection in the prompt selection instead of
        # using your keyboard
        invalidKeyError  = [
            {
                'type': 'list',
                'name': 'key_error',
                'message': 'Please only use your keyboard and the enter key when making a selection.',
                'choices': [
                    {
                        'name': 'I would like to try again',
                    },
                    {
                        'name': 'Exit the app',
                    }
                ],
            }
        ]
        key_error = prompt(invalidKeyError)
        if key_error != {} and key_error["key_error"] == 'I would like to try again':
            authorization()
        else:
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
                },
                {
                    'name': 'Transcription + Translation Only',
                },
                {
                    'name': 'Exit the app',
                }
            ],
        }
    ]
    process_selected = prompt(translation_transcription_prompt)
    if process_selected["process_selected"] == 'Exit the app':
        return
    # Input file
    rprint("[blue]=============================[blue]")
    rprint(f"[bold]Enter the absolute path to the audio file you want to do the \"{process_selected['process_selected']}\" process on:[bold]")
    while True:
        input_file = input()
        input_audio_path = input_file.strip()  # remove leading and trailing whitespace
        # Check 1 for input file: Validate whether path is valid
        audio_path_last_backslash_index = input_file.rfind("/")
        audio_name = input_file[audio_path_last_backslash_index + 1:]
        audio_name = audio_name.strip()  # remove leading and trailing whitespace
        # TODO: Purposefully left out the "replace <spaces> with <"_"> since that might interfere with the path checking
        input_audio_path = (input_audio_path.strip())[0: input_audio_path.rfind("/") + 1] + audio_name
        print("Revised input audio path: ", input_audio_path)
        is_valid_audio_path = validate_path(input_audio_path, True)
        if is_valid_audio_path:
            break
        else:
            print("You entered an invalid audio path. Please try again")


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
                {
                    'name': 'Exit the app',
                },
            ],
        }
    ]
    to_english_selection = prompt(to_eng_selection_prompt)
    if to_english_selection["to_english_selection"] == 'Exit the app':
        return

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
                {
                    'name': 'Exit the app',
                },
            ],
        }
    ]
    model_size_selection = prompt(model_size_selection_prompt)
    if model_size_selection["model_size_selection"] == 'Exit the app':
        return
    rprint("[blue]=============================[blue]")
    # Destination Folder
    rprint(f"[bold]Enter the absolute path to your destination folder:[bold]")
    while True:
        destination_selection = input()
        destination_selection = destination_selection.strip() # remove leading and trailing whitespace
        is_valid_dest_path = validate_path(destination_selection, False)
        if is_valid_dest_path:
            break
        else:
            print("You entered an invalid destination path. Please try again")

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

#TODO: Need to complete this method later
# How to use (previously: help) section
# (note: this is different from the option --help, which list out all the options the user can use)
# Called by using the option -howtouse
def howtouse_ui():
    rprint("[magenta]=============================[magenta]")
    rprint("[bold][underline]STREET Lab Whisper App[underline][bold]")
    rprint("")
    rprint("[bold]How to Use[bold]")
    rprint("[magenta]=============================[magenta]")

# Credits section
# Called by using the option -credits
def credits_ui():
    rprint("[magenta]=============================[magenta]")
    rprint("[bold][underline]STREET Lab Whisper App[underline][bold]")
    rprint("")
    rprint("[bold]Credits[bold]")
    rprint("This application was created by STREET Lab: https://www.streetlab.tech/ ")
    rprint("For details about the technologies and libraries used, visit the following repository: https://github.com/moonsdust/street-whisper-app")
    rprint("[magenta]=============================[magenta]")

if __name__ == "__main__":
    typer.run(startup_ui)
