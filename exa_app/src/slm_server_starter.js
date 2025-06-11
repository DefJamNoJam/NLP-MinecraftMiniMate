/**
 * slm_server_starter.js  -  Stub version
 * --------------------------------------
 * 외부에서 이미 실행 중인 RAG(FASTAPI) 서버를 사용한다고 가정하고,
 * Electron 쪽에서는 서버 제어(start/stop)를 시도하지 않도록 만든 더미 모듈.
 */

function log(msg) {
  console.log(`[SLM-STUB] ${msg}`);
}

// 최초 require 시 안내 로그
log('External RAG server mode - Electron will NOT start/stop the backend.');

// Always report “running”
function isRunning() {
  return true;
}

// startServer() → 즉시 성공 리턴
async function startServer() {
  log('startServer() called - skipping (external server assumed).');
  return true;
}

// stopServer() → 아무 것도 하지 않음
function stopServer() {
  log('stopServer() called - skipping (external server assumed).');
}

// 내보내기
module.exports = {
  startServer,
  stopServer,
  isRunning
};
