#!/usr/bin/env python3
"""
KrishiMitra Project Structure Setup Script
Run this script in your project root directory to create the complete folder structure.
"""

import os

def create_project_structure():
    # Define the project structure
    structure = {
        'ML': {
            'models': ['leaf_disease_detector.py', 'soil_analyzer.py', 'voice_processor.py'],
            'ml_models': {
                'leaf_disease': ['labels.txt'],
                'soil_analysis': ['soil_types.json']
            },
            'bridge': ['unity_bridge.py', 'api_server.py'],
            'data': {
                'training_data': {
                    'diseases': [],
                    'soil_types': []
                },
                'test_images': []
            },
            'scripts': ['train_models.py', 'data_preprocessing.py', 'model_converter.py']
        },
        'Unity': {
            'unity_app': {
                'Assets': {
                    'Scripts': ['MLManager.cs', 'CameraController.cs', 'VoiceController.cs', 'ARController.cs', 'UIManager.cs'],
                    'Scenes': [],
                    'StreamingAssets': [],
                    'Prefabs': [],
                    'Resources': {
                        'Textures': [],
                        'Audio': []
                    }
                },
                'ProjectSettings': []
            },
            'builds': []
        },
        'assets': {
            'icons': [],
            'images': [],
            'sounds': []
        },
        'audio_responses': {
            'hindi': {
                'disease_responses': [],
                'soil_responses': [],
                'general_responses': []
            },
            'marathi': {
                'disease_responses': [],
                'soil_responses': [],
                'general_responses': []
            },
            'tamil': {
                'disease_responses': [],
                'soil_responses': [],
                'general_responses': []
            },
            'telugu': {
                'disease_responses': [],
                'soil_responses': [],
                'general_responses': []
            },
            'english': {
                'disease_responses': [],
                'soil_responses': [],
                'general_responses': []
            }
        },
        'docs': ['setup_guide.md', 'api_documentation.md', 'user_manual.md'],
        'tools': {
            'deployment_scripts': [],
            '': ['dataset_builder.py', 'model_tester.py']
        }
    }

    def create_structure(base_path, structure_dict):
        for key, value in structure_dict.items():
            if key == '':  # Handle files in the current directory
                for file_name in value:
                    file_path = os.path.join(base_path, file_name)
                    create_file(file_path)
            else:
                folder_path = os.path.join(base_path, key)
                os.makedirs(folder_path, exist_ok=True)
                print(f"Created folder: {folder_path}")
                
                if isinstance(value, dict):
                    create_structure(folder_path, value)
                elif isinstance(value, list):
                    for file_name in value:
                        file_path = os.path.join(folder_path, file_name)
                        create_file(file_path)

    def create_file(file_path):
        try:
            with open(file_path, 'w') as f:
                if file_path.endswith('.py'):
                    f.write('# TODO: Implement this module\n')
                elif file_path.endswith('.cs'):
                    f.write('// TODO: Implement this Unity script\n')
                elif file_path.endswith('.md'):
                    f.write('# TODO: Add documentation\n')
                elif file_path.endswith('.txt'):
                    f.write('# TODO: Add labels/data\n')
                elif file_path.endswith('.json'):
                    f.write('{\n  "TODO": "Add data structure"\n}\n')
                else:
                    f.write('')
            print(f"Created file: {file_path}")
        except Exception as e:
            print(f"Error creating file {file_path}: {e}")

    # Start creating the structure
    print("Creating KrishiMitra project structure...")
    print("=" * 50)
    
    create_structure('.', structure)
    
    print("=" * 50)
    print("✅ Project structure created successfully!")
    print("\nNext steps:")
    print("1. Start developing ML models in ML/models/")
    print("2. Train and save models to ML/ml_models/")
    print("3. Develop Unity application in Unity/unity_app/")
    print("4. Test integration between ML and Unity components")

if __name__ == "__main__":
    create_project_structure()