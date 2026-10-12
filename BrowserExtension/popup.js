async function refresh(type) {
  const response = await chrome.runtime.sendMessage({type});
  document.querySelector('#status').textContent = response?.status || '连接不可用';
}
document.querySelector('#retry').onclick = () => refresh('reconnect');
refresh('status');
