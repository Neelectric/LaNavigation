console.log('home.js loaded');
let mediaRecorder;
        let audioChunks = [];

        document.getElementById('btnStart').addEventListener('click', async function () {
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

            // Enable/Disable buttons
            document.getElementById('btnStart').disabled = true;
            document.getElementById('btnStop').disabled = false;
        });

        document.getElementById('btnStop').addEventListener('click', function () {
            mediaRecorder.stop();
            console.log('Stopped recording.');
            document.getElementById('btnStart').disabled = false;
            document.getElementById('btnStop').disabled = true;
        });

        function sendAudioToBackend(audioBlob) {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'audio.wav');

            // AJAX request to Django backend
            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/record/', true);
            xhr.setRequestHeader('X-CSRFToken', getCookie('csrftoken'));  // For CSRF protection
            xhr.send(formData);

            xhr.onload = function () {
                if (xhr.status === 200) {
                    console.log('Audio uploaded successfully');
                } else {
                    console.error('Error uploading audio');
                }
            };
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