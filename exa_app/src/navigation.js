/**
 * navigation.js
 * - 페이드 전환 효과
 * - 히스토리 스택 기반 뒤로가기
 * - 뒤로가기 버튼 활성화/비활성화
 */

// 전역 변수 설정
const TRANSITION_TIME = 300; // ms (styles.css의 .3s와 일치)
let navigationHistory = []; // 방문 기록 저장 배열

// 히스토리 초기화 함수
function initializeHistory() {
  const currentPage = window.location.pathname.split('/').pop() || 'index.html';
  if (navigationHistory.length === 0) {
    navigationHistory.push(currentPage);
    console.log('Initialized navigation history with current page:', currentPage);
  }
}

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
  console.log('Navigation.js loaded');
  
  // 히스토리 초기화
  initializeHistory();
  
  // 현재 페이지 경로 가져오기
  const currentPage = window.location.pathname.split('/').pop() || 'index.html';
  console.log('Current page:', currentPage);
  
  // 항상 현재 페이지를 히스토리에 추가 (이전에 없던 경우에만)
  if (navigationHistory[navigationHistory.length - 1] !== currentPage) {
    navigationHistory.push(currentPage);
    console.log('Added current page to navigation history:', navigationHistory);
  }
  
  // 페이드인 효과 적용
  document.body.classList.add('fade-transition');
  setTimeout(() => {
    document.body.classList.remove('fade-transition');
  }, TRANSITION_TIME);
  
  // 뒤로가기 버튼 이벤트 리스너 설정
  setupBackButton();
  
  // 뒤로가기 버튼 상태 업데이트
  updateBackButtonState();
});

// 뒤로가기 버튼 설정
function setupBackButton() {
  console.log('Setting up back button...');
  const backButton = document.getElementById('global-back');
  
  if (backButton) {
    console.log('Back button found, adding event listeners');
    
    // 기존 이벤트 리스너 제거 (중복 방지)
    backButton.removeEventListener('click', handleBackNavigation);
    
    // 새 이벤트 리스너 추가
    backButton.addEventListener('click', handleBackNavigation);
    console.log('Back button event listener attached');
    
    // 직접 onclick 속성 설정 (이중 보장)
    backButton.onclick = handleBackNavigation;
    
    // 버튼이 클릭 가능한지 확인
    console.log('Back button onclick handler:', backButton.onclick);
    console.log('Back button disabled status:', backButton.disabled);
  } else {
    console.warn('Back button not found in the current page');
    // 페이지의 모든 버튼 로깅 (디버깅용)
    console.log('All buttons on page:', document.querySelectorAll('button'));
  }
}

// 뒤로가기 버튼 상태 업데이트
function updateBackButtonState() {
  const backButton = document.getElementById('global-back');
  
  if (!backButton) {
    console.warn('Back button not found in updateBackButtonState');
    return;
  }
  
  // 항상 뒤로가기 버튼을 활성화 (비활성화하지 않음)
  backButton.classList.remove('disabled');
  backButton.disabled = false;
  console.log('Back button enabled (always)');
  
  // 디버깅을 위해 히스토리 상태 로깅
  console.log('Current navigation history:', navigationHistory);
}

// 뒤로가기 처리 함수
function handleBackNavigation(event) {
  console.log('=== handleBackNavigation called ===');
  
  // 이벤트 기본 동작 방지
  if (event) {
    event.preventDefault();
    event.stopPropagation(); // 이벤트 버블링 방지
  }
  
  const currentPage = window.location.pathname.split('/').pop() || 'index.html';
  console.log('Back button clicked - current page:', currentPage);
  console.log('Current navigation history:', navigationHistory);
  
  // settings 페이지에서 뒤로가기를 누른 경우
  if (currentPage === 'settings.html') {
    console.log('Back from settings page');
    
    // 히스토리에서 settings.html 제거 (현재 페이지)
    if (navigationHistory.length > 0 && navigationHistory[navigationHistory.length - 1] === 'settings.html') {
      navigationHistory.pop();
    }
    
    // 이전 페이지가 있으면 그 페이지로, 없으면 인증 상태에 따라 기본 페이지로
    if (navigationHistory.length > 0) {
      // 이전 페이지로 이동 (히스토리에서 제거하지 않음)
      const previousPage = navigationHistory[navigationHistory.length - 1];
      console.log('Navigating back to previous page:', previousPage);
      
      // 페이드아웃 효과 적용
      document.body.classList.add('fade-out');
      
      // 전환 후 페이지 이동
      setTimeout(() => {
        if (window.electronAPI && window.electronAPI.navigateTo) {
          window.electronAPI.navigateTo(previousPage, false);
        } else {
          window.location.href = previousPage;
        }
      }, TRANSITION_TIME);
    } else {
      // 히스토리가 없는 경우
      console.log('No navigation history, going to launch page');
      
      // 페이드아웃 효과 적용
      document.body.classList.add('fade-out');
      
      // 전환 후 launch 페이지로 이동
      setTimeout(() => {
        if (window.electronAPI && window.electronAPI.navigateTo) {
          window.electronAPI.navigateTo('launch.html', false);
        } else {
          window.location.href = 'launch.html';
        }
      }, TRANSITION_TIME);
    }
    return;
  }
  
  // 일반적인 뒤로가기 처리 (settings 페이지가 아닌 경우)
  if (navigationHistory.length > 0) {
    // 현재 페이지를 히스토리에서 제거 (마지막 항목이 현재 페이지와 같은 경우)
    if (navigationHistory[navigationHistory.length - 1] === currentPage) {
      navigationHistory.pop();
      console.log('Removed current page from history');
    }
    
    // 이전 페이지로 이동 (히스토리에서 제거하지 않고 확인만 함)
    const previousPage = navigationHistory[navigationHistory.length - 1] || 'index.html';
    console.log('Navigating back to:', previousPage);
    
    // 페이드아웃 효과 적용
    document.body.classList.add('fade-out');
    
    // 전환 후 페이지 이동
    setTimeout(() => {
      if (window.electronAPI && window.electronAPI.navigateTo) {
        window.electronAPI.navigateTo(previousPage, false)
          .then(result => {
            console.log('Back navigation successful:', result);
          })
          .catch(error => {
            console.error('Back navigation failed:', error);
            // 실패 시 현재 페이지를 히스토리에 다시 추가
            navigationHistory.push(currentPage);
          });
      } else {
        console.error('electronAPI.navigateTo not available');
        window.location.href = previousPage;
      }
    }, TRANSITION_TIME);
  } else {
    // 히스토리가 없는 경우 기본 페이지로 이동
    console.log('No history, going to index.html');
    if (window.electronAPI && window.electronAPI.navigateTo) {
      window.electronAPI.navigateTo('index.html', false);
    } else {
      window.location.href = 'index.html';
    }
  }
}

// 키보드 단축키 설정
window.addEventListener('keydown', (event) => {
  // Alt+← 또는 Esc 키로 뒤로가기
  if ((event.altKey && event.key === 'ArrowLeft') || event.key === 'Escape') {
    handleBackNavigation();
  }
});

// electronAPI.navigateTo 함수 래핑 (있는 경우)
if (window.electronAPI && window.electronAPI.navigateTo) {
  const originalNavigate = window.electronAPI.navigateTo;
  
  window.electronAPI.navigateTo = (page, addToHistory = true) => {
    console.log(`Navigating to ${page}, addToHistory: ${addToHistory}`);
    
    // 현재 페이지 가져오기
    const currentPage = window.location.pathname.split('/').pop();
    
    // 페이드아웃 효과 적용
    document.body.classList.add('fade-out');
    
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        // 히스토리에 추가 (필요한 경우)
        if (addToHistory) {
          // 현재 페이지가 이미 히스토리의 마지막 항목이 아니면 추가
          if (navigationHistory.length === 0 || navigationHistory[navigationHistory.length - 1] !== currentPage) {
            navigationHistory.push(currentPage);
          }
          console.log('Updated navigation history:', navigationHistory);
        }
        
        // 원래 함수 호출
        originalNavigate(page)
          .then(result => {
            console.log('Navigation successful:', result);
            // 페이지 이동 후 뒤로가기 버튼 상태 업데이트
            updateBackButtonState();
            resolve(result);
          })
          .catch(error => {
            console.error('Navigation failed:', error);
            // 실패 시 페이드아웃 효과 제거
            document.body.classList.remove('fade-out');
            reject(error);
          });
      }, TRANSITION_TIME);
    });
  };
  
  console.log('electronAPI.navigateTo wrapped successfully');
} else {
  console.warn('electronAPI.navigateTo not found - navigation may not work properly');
}

// 디버깅용 전역 접근 제공
window.debugNavigation = {
  getHistory: () => navigationHistory,
  clearHistory: () => {
    navigationHistory = [window.location.pathname.split('/').pop()];
    updateBackButtonState();
    return 'History cleared except current page';
  },
  forceBack: function() {
    console.log('forceBack called from debugNavigation');
    // 히스토리가 없으면 기본 페이지로 이동
    if (navigationHistory.length <= 1) {
      console.log('No history, going to index.html');
      if (window.electronAPI && window.electronAPI.navigateTo) {
        return window.electronAPI.navigateTo('index.html');
      } else {
        window.location.href = 'index.html';
      }
    } else {
      // 일반적인 뒤로가기 처리
      return handleBackNavigation({ preventDefault: () => {} });
    }
  }
};

// 테스트용 내보내기 (Node.js 환경)
if (typeof module !== 'undefined') {
  module.exports = {
    handleBackNavigation,
    navigationHistory,
    updateBackButtonState
  };
}
