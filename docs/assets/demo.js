'use strict';
const audio = document.getElementById('input-audio');
const speed = document.getElementById('playback-speed');
const error = document.getElementById('audio-error');
speed.addEventListener('change', () => { audio.playbackRate = Number(speed.value); });
audio.addEventListener('error', () => { error.hidden = false; });
audio.addEventListener('canplay', () => { error.hidden = true; });
