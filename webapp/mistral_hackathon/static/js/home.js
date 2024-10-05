document.addEventListener('DOMContentLoaded', function () {
    const messageList = document.getElementById('messageList');
    let messageCount = 0;

    function addMessage(owner, content) {
        // Implementation to add message to your UI
        console.log(`New message from ${owner}: ${content}`);

        // Update UI
        const messageDiv = document.createElement('div');
        messageDiv.className = owner === 'input' ? 'user-message' : 'model-message';
        messageDiv.textContent = content;
        messageList.appendChild(messageDiv);
        messageDiv.scrollIntoView({ behavior: 'smooth' });
        /*
        event.preventDefault();  // Prevent full page reload

            const textInput = document.getElementById('text-input').value;
            const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;

            // Make a POST request using Fetch API
            fetch("{% url 'process_text' %}", {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'X-CSRFToken': csrfToken
                },
                body: new URLSearchParams({
                    'text_input': textInput
                })
            })
            .then(response => response.json())
            .then(data => {
                // Set the audio source to the generated file and play it
                const audioPlayer = document.getElementById('audioPlayer');
                const audioSource = document.getElementById('audioSource');
                
                audioSource.src = data.audio_url;
                audioPlayer.style.display = 'block';  // Show the audio player
                audioPlayer.load();  // Load the new audio
                audioPlayer.play();  // Play the audio
            })
            .catch(error => console.error('Error:', error));
        };*/


console.log('home.js loaded');
let mediaRecorder;
let audioChunks = [];
let recording = false;


document.getElementById('inputBtn').addEventListener('click', async function () {
    console.log('Button clicked');
    if (!recording) {
        console.log('Starting recording...');
        recording = true;
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.start();
        console.log('Recording started...');

        mediaRecorder.ondataavailable = function (event) {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = function () {
            console.log('Recording stopped.');
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            sendAudioToBackend(audioBlob);
        };
    } else {
        recording = false;
    mediaRecorder.stop();
    console.log('Stopped recording.');
    }
});

function sendAudioToBackend(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'audio.wav');

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/record/', true);
    xhr.setRequestHeader('X-CSRFToken', getCookie('csrftoken'));  // For CSRF protection
    xhr.send(formData);
    console.log(xhr.response);
    xhr.onload = function () {
        if (xhr.status === 200) {
            console.log('Audio uploaded successfully');
            const response = JSON.parse(xhr.responseText);
            
            // Check if there are new messages and add them
            if (response.new_messages) {
                response.new_messages.forEach(message => {
                    addMessage(message.owner, message.content);
                });
            }
        } else {
            console.error('Error uploading audio');
        }
    };

    xhr.send(formData);

    fetch('/home/', {
        method: 'POST',
        body: formData  // Assuming this contains the file
    }).then(response => response.json())
    .then(data => {
        const audioId = data.audio_id;
        // Display user's input message
        displayMessage(data.new_messages);

        // Start polling for the answer
        const pollInterval = setInterval(() => {
            fetch(`/get-answer/${audioId}/`)
            .then(response => response.json())
            .then(answerData => {
                if (answerData.status === 'success') {
                    // Answer is ready, display it
                    displayMessage([answerData.answer_message]);
                    clearInterval(pollInterval);  // Stop polling once the answer is received
                }
            });
        }, 3000);  // Poll every 3 seconds
    });
}

function changeLoading() {
    inputButton.disabled = isLoading;
    if (isLoading) {
        addMessage('system', 'Processing...');
    }
}

// Function to get CSRF token from cookies
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}



}});