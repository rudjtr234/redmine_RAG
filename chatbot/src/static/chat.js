// ============================================================
// MTS AI-DT 챗봇 v0.4.0 — chat.js
// ============================================================

// 전역 상태
const chatBox = document.getElementById('chat-box');
const messagesInner = chatBox.querySelector('.messages-inner');
const questionInput = document.getElementById('question-input');
const sendBtn = document.getElementById('send-btn');

let currentAbortController = null;
let currentLoadingMessage = null;
let activeConversationId = null;
let activeDiagramCount = 0;
// 진행 중인 도식화 취소 핸들러 목록 { cancel: fn }
const activeDiagrams = [];

// ============================================================
// 사이드바 토글
// ============================================================
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebarOverlay = document.getElementById('sidebar-overlay');

function isMobile() { return window.innerWidth <= 768; }

sidebarToggle.addEventListener('click', () => {
    if (isMobile()) {
        const isOpen = sidebar.classList.contains('mobile-open');
        sidebar.classList.toggle('mobile-open', !isOpen);
        sidebarOverlay.classList.toggle('visible', !isOpen);
    } else {
        const isCollapsed = sidebar.classList.contains('collapsed');
        sidebar.classList.toggle('collapsed', !isCollapsed);
        sidebarToggle.classList.toggle('collapsed-pos', !isCollapsed);
        localStorage.setItem('sidebarCollapsed', !isCollapsed ? '1' : '0');
    }
});
sidebarOverlay.addEventListener('click', () => {
    sidebar.classList.remove('mobile-open');
    sidebarOverlay.classList.remove('visible');
});

if (!isMobile() && localStorage.getItem('sidebarCollapsed') === '1') {
    sidebar.classList.add('collapsed');
    sidebarToggle.classList.add('collapsed-pos');
}

// ============================================================
// 유틸리티
// ============================================================
function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function convertMarkdownTable(text) {
    // 1. 이미지 마크다운 플레이스홀더 추출 (/redmine-image/숫자 경로만 허용)
    const imgPlaceholders = [];
    const IMG_ALLOW_RE = /!\[([^\]]*)\]\((\/redmine-image\/(\d+))\)/g;
    text = text.replace(IMG_ALLOW_RE, (_, alt, src) => {
        const idx = imgPlaceholders.length;
        imgPlaceholders.push({ alt, src });
        return `%%IMG_${idx}%%`;
    });
    // raw <img> 태그 차단 (allowlist 외 HTML 이미지)
    text = text.replace(/<img[^>]*>/gi, '');

    // 2. 마크다운 기본 문법 변환
    // bold: **텍스트**
    text = text.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
    // heading: ## ~ ####
    text = text.replace(/^#{2,4}\s+(.+)$/gm, '<h3 style="margin:12px 0 6px;font-size:1em;">$1</h3>');
    // unordered list: 연속 "- " 항목
    text = text.replace(/((?:^- .+\n?)+)/gm, match => {
        const items = match.trim().split('\n')
            .map(l => `<li>${l.replace(/^- /, '')}</li>`).join('');
        return `<ul style="margin:6px 0;padding-left:20px;">${items}</ul>`;
    });
    // 링크 allowlist: /issues/{숫자} 또는 https://redmine. 도메인만
    text = text.replace(
        /\[([^\]]+)\]\(((?:\/issues\/\d+|https?:\/\/redmine\.[^\s)]+))\)/g,
        '<a href="$2" target="_blank" rel="noopener">$1</a>'
    );

    // 3. 표 처리 (기존 로직 유지)
    const lines = text.split('\n');
    let html = text;
    let inTable = false;
    let tableLines = [];
    const processTable = (tlines) => {
        let tableHtml = '<table border="1" style="border-collapse:collapse;margin:10px 0;width:100%;">';
        tlines.forEach((line, idx) => {
            if (line.includes('---')) return;
            const cells = line.split('|').filter(cell => cell.trim() !== '');
            const tag = idx === 0 ? 'th' : 'td';
            const style = idx === 0
                ? 'style="background:#f0f0f0;padding:8px;text-align:left;"'
                : 'style="padding:8px;"';
            tableHtml += '<tr>';
            cells.forEach(cell => tableHtml += `<${tag} ${style}>${cell.trim()}</${tag}>`);
            tableHtml += '</tr>';
        });
        return tableHtml + '</table>';
    };
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line.startsWith('|') && line.endsWith('|')) {
            if (!inTable) { inTable = true; tableLines = []; }
            tableLines.push(line);
        } else if (inTable && tableLines.length > 0) {
            html = html.replace(tableLines.join('\n'), processTable(tableLines));
            inTable = false; tableLines = [];
        }
    }
    if (inTable && tableLines.length > 0) {
        html = html.replace(tableLines.join('\n'), processTable(tableLines));
    }

    // 4. 줄바꿈
    html = html.replace(/\n/g, '<br>');

    // 5. 이미지 플레이스홀더 복원 (줄바꿈 변환 후 <br> 후행 제거 포함)
    html = html.replace(/%%IMG_(\d+)%%(<br>)?/g, (_, idx) => {
        const { alt, src } = imgPlaceholders[parseInt(idx)];
        const safeAlt = alt.replace(/[<>"'&]/g,
            c => ({ '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;', '&': '&amp;' }[c]));
        return `<span class="answer-inline-img-wrap">` +
            `<img src="${src}" alt="${safeAlt}" class="answer-inline-img"` +
            ` loading="lazy" decoding="async"` +
            ` onerror="this.closest('.answer-inline-img-wrap').innerHTML=` +
            `'<span class=\\'img-load-fail\\'>이미지 로드 실패&nbsp;&middot;&nbsp;` +
            `<a href=\\'${src}\\' target=\\'_blank\\'>링크</a></span>'">` +
            (safeAlt ? `<span class="answer-inline-caption">${safeAlt}</span>` : '') +
            `</span>`;
    });

    return html;
}

function closeDiagramDownloadDropdown(selectEl) {
    if (!selectEl) return;
    selectEl.classList.remove('is-open');
    selectEl.setAttribute('aria-expanded', 'false');
    const trigger = selectEl.querySelector('.diagram-download-format-trigger');
    if (trigger) trigger.setAttribute('aria-expanded', 'false');
}
function closeAllDiagramDownloadDropdowns() {
    document.querySelectorAll('.diagram-download-format-select.is-open').forEach(el => closeDiagramDownloadDropdown(el));
}
document.addEventListener('click', (e) => {
    if (!e.target.closest('.diagram-download-format-select')) closeAllDiagramDownloadDropdowns();
    if (!e.target.closest('.msg-more-wrap')) {
        document.querySelectorAll('.msg-more-menu.open').forEach(m => m.classList.remove('open'));
    }
});
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeAllDiagramDownloadDropdowns();
        document.querySelectorAll('.msg-more-menu.open').forEach(m => m.classList.remove('open'));
    }
});

// ============================================================
// 사용자 이름 관리
// ============================================================
async function initServerSession(userName) {
    try {
        const resp = await fetch('/session/init', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_name: userName })
        });
        if (resp.ok) {
            const data = await resp.json();
            const isAdmin = data.is_admin === true;
            sessionStorage.setItem('isAdmin', isAdmin ? '1' : '0');
            _applyAdminUI(isAdmin);
        }
    } catch (e) { console.warn('세션 초기화 실패:', e); }
}

function _applyAdminUI(isAdmin) {
    const btn = document.getElementById('user-list-btn');
    if (btn) btn.style.display = isAdmin ? '' : 'none';
}

async function setUserName() {
    const nameInput = document.getElementById('userNameInput');
    const userName = nameInput.value.trim();
    if (!userName) { alert('이름을 입력해주세요'); return; }
    const cleanName = userName.replace(/[^a-zA-Z0-9가-힣]/g, '');
    if (cleanName !== userName) { alert('특수문자는 사용할 수 없습니다'); return; }
    const previousName = sessionStorage.getItem('userName');
    sessionStorage.setItem('userName', cleanName);
    document.getElementById('nameModal').style.display = 'none';
    updateSidebarUser(cleanName);
    // 서버 세션에 user_name 즉시 등록 (이미지 프록시 인증용)
    await initServerSession(cleanName);
    if (!previousName || previousName !== cleanName) {
        resetConversationUI();
        resetServerSession();
        activeConversationId = null;
    }
    loadConversationList(cleanName);
    questionInput.focus();
}

function changeUserName() {
    sessionStorage.removeItem('userName');
    sessionStorage.removeItem('isAdmin');
    _applyAdminUI(false);  // 관리자 버튼 즉시 숨김
    activeConversationId = null;
    document.getElementById('nameModal').style.display = 'flex';
    document.getElementById('userNameInput').value = '';
    document.getElementById('userNameInput').focus();
    document.getElementById('sidebar-user-row').style.display = 'none';
    document.getElementById('conv-list').innerHTML = '<div class="conv-loading">불러오는 중...</div>';
    resetConversationUI();
    resetServerSession();
}

function updateSidebarUser(userName) {
    document.getElementById('sidebar-user-name').textContent = userName;
    document.getElementById('sidebar-user-row').style.display = 'flex';
}

// welcome-box 원본 HTML을 페이지 로드 시 한 번 저장
const _welcomeBoxHTML = (() => {
    const el = document.getElementById('welcome-msg');
    return el ? el.outerHTML : '';
})();

function resetConversationUI() {
    messagesInner.innerHTML = _welcomeBoxHTML;
    renderSourcesPanel([]);
}

async function resetServerSession(conversation_id = null) {
    try {
        const body = conversation_id ? { conversation_id } : {};
        await fetch('/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
    } catch (e) {
        console.warn('세션 초기화 실패(무시):', e);
    }
}

// ============================================================
// 대화 목록 (사이드바)
// ============================================================
async function loadConversationList(userName) {
    const list = document.getElementById('conv-list');
    list.innerHTML = '<div class="conv-loading">불러오는 중...</div>';
    try {
        const resp = await fetch(`/conversations?user_name=${encodeURIComponent(userName)}`);
        const data = await resp.json();
        renderConvList(data.conversations || [], userName);
    } catch (e) {
        list.innerHTML = '<div class="conv-empty">불러오기 실패</div>';
    }
}

function getConvTitle(convId, fallback) {
    return localStorage.getItem(`convTitle_${convId}`) || fallback || '새 대화';
}

function saveConvTitle(convId, title) {
    localStorage.setItem(`convTitle_${convId}`, title);
}

// ── 컨텍스트 메뉴 ──────────────────────────────────────────────
let _activeCtxMenu = null;

function closeCtxMenu() {
    if (_activeCtxMenu) { _activeCtxMenu.remove(); _activeCtxMenu = null; }
}
document.addEventListener('click', closeCtxMenu);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeCtxMenu(); });

function showConvCtxMenu(e, item, convId, userName) {
    e.stopPropagation();
    closeCtxMenu();

    const menu = document.createElement('div');
    menu.className = 'conv-ctx-menu';
    _activeCtxMenu = menu;

    const renameItem = document.createElement('div');
    renameItem.className = 'conv-ctx-item';
    renameItem.innerHTML = '<span>✏️</span> 이름 변경';
    renameItem.addEventListener('click', (ev) => {
        ev.stopPropagation();
        closeCtxMenu();
        startConvTitleEdit(item, convId);
    });

    const deleteItem = document.createElement('div');
    deleteItem.className = 'conv-ctx-item danger';
    deleteItem.innerHTML = '<span>🗑️</span> 삭제';
    deleteItem.addEventListener('click', (ev) => {
        ev.stopPropagation();
        closeCtxMenu();
        deleteConversationUI(convId, item, userName);
    });

    menu.appendChild(renameItem);
    menu.appendChild(deleteItem);
    // 숨긴 채로 삽입해 크기 측정 후 위치 결정
    menu.style.visibility = 'hidden';
    document.body.appendChild(menu);

    const rect = e.currentTarget.getBoundingClientRect();
    const mw = menu.offsetWidth;
    const mh = menu.offsetHeight;
    let top = rect.bottom + 4;
    let left = rect.right - mw;
    if (top + mh > window.innerHeight) top = rect.top - mh - 4;
    if (left < 4) left = 4;
    menu.style.top = top + 'px';
    menu.style.left = left + 'px';
    menu.style.visibility = '';
}

function getDeletedConvIds() {
    try { return JSON.parse(localStorage.getItem('deletedConvIds') || '[]'); } catch { return []; }
}
function markConvDeleted(convId) {
    const ids = getDeletedConvIds();
    if (!ids.includes(convId)) { ids.push(convId); localStorage.setItem('deletedConvIds', JSON.stringify(ids)); }
}

async function deleteConversationUI(convId, item, userName) {
    // 서버 DB에서 실제 삭제
    try {
        const resp = await fetch(`/conversations/${convId}?user_name=${encodeURIComponent(userName)}`, { method: 'DELETE' });
        if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));
            console.warn('대화 삭제 실패:', data.error || resp.status);
        }
    } catch (e) {
        console.warn('대화 삭제 요청 오류:', e);
    }

    // UI 정리
    markConvDeleted(convId);
    localStorage.removeItem(`convTitle_${convId}`);
    localStorage.removeItem(`activeConversationId_${userName}`);

    item.remove();

    // 삭제된 게 현재 활성 대화면 화면 초기화
    if (activeConversationId === convId) {
        activeConversationId = null;
        // 다른 대화 있으면 첫 번째로 이동, 없으면 빈 화면
        const next = document.querySelector('.conv-item');
        if (next) {
            selectConversation(next.dataset.convId, userName);
        } else {
            resetConversationUI();
            document.getElementById('conv-list').innerHTML =
                '<div class="conv-empty">대화 내역이 없습니다.</div>';
        }
    }
}

// ── 인라인 제목 편집 ──────────────────────────────────────────
function startConvTitleEdit(item, convId) {
    const titleEl = item.querySelector('.conv-item-title');
    if (!titleEl || titleEl.querySelector('input')) return;
    const currentTitle = item.dataset.displayTitle || titleEl.textContent.trim();
    titleEl.textContent = '';
    const input = document.createElement('input');
    input.type = 'text';
    input.value = currentTitle;
    input.className = 'conv-title-input';
    input.maxLength = 50;
    const commit = () => {
        const newTitle = input.value.trim() || currentTitle;
        titleEl.textContent = newTitle;
        item.title = newTitle;
        item.dataset.displayTitle = newTitle;
        saveConvTitle(convId, newTitle);
    };
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
        if (e.key === 'Escape') { input.value = currentTitle; input.blur(); }
        e.stopPropagation();
    });
    input.addEventListener('blur', commit);
    input.addEventListener('click', (e) => e.stopPropagation());
    titleEl.appendChild(input);
    input.focus();
    input.select();
}

// ── conv-item 생성 공통 헬퍼 ──────────────────────────────────
function createConvItem(convId, displayTitle, userName, isActive) {
    const item = document.createElement('div');
    item.className = 'conv-item' + (isActive ? ' active' : '');
    item.dataset.convId = convId;
    item.dataset.displayTitle = displayTitle;
    item.title = displayTitle;

    const titleEl = document.createElement('span');
    titleEl.className = 'conv-item-title';
    titleEl.textContent = displayTitle;

    const menuBtn = document.createElement('button');
    menuBtn.className = 'conv-item-menu-btn';
    menuBtn.title = '더 보기';
    menuBtn.textContent = '⋯';
    menuBtn.addEventListener('click', (e) => showConvCtxMenu(e, item, convId, userName));

    item.appendChild(titleEl);
    item.appendChild(menuBtn);
    item.addEventListener('click', () => selectConversation(convId, userName));
    return item;
}

function renderConvList(convs, userName) {
    const list = document.getElementById('conv-list');
    list.innerHTML = '';
    const deleted = getDeletedConvIds();
    const visible = convs.filter(c => !deleted.includes(c.conversation_id));
    if (visible.length === 0) {
        list.innerHTML = '<div class="conv-empty">대화 내역이 없습니다.</div>';
        return;
    }
    visible.forEach(conv => {
        const displayTitle = getConvTitle(conv.conversation_id, conv.title);
        const item = createConvItem(conv.conversation_id, displayTitle, userName, conv.conversation_id === activeConversationId);
        list.appendChild(item);
    });
}

async function selectConversation(convId, userName) {
    if (convId === activeConversationId) return;
    if (activeDiagrams.length > 0) {
        if (!confirm(`도식화가 진행 중입니다. 대화를 전환하면 도식화가 취소됩니다.\n계속하시겠습니까?`)) return;
        // 진행 중인 도식화 전부 취소
        [...activeDiagrams].forEach(h => h.cancel());
    }
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
        if (currentLoadingMessage) { removeLoadingPlaceholder(currentLoadingMessage); currentLoadingMessage = null; }
        setButtonState(false);
    }
    activeConversationId = convId;
    localStorage.setItem(`activeConversationId_${userName}`, convId);
    document.querySelectorAll('.conv-item').forEach(el => {
        el.classList.toggle('active', el.dataset.convId === convId);
    });
    messagesInner.innerHTML = '';
    renderSourcesPanel([]);
    const loading = addLoadingPlaceholder();
    try {
        const resp = await fetch(`/conversations/${encodeURIComponent(convId)}?user_name=${encodeURIComponent(userName)}`);
        const data = await resp.json();
        removeLoadingPlaceholder(loading);
        const messages = data.messages || [];
        if (messages.length === 0) {
            resetConversationUI();
        } else {
            messagesInner.innerHTML = '';
            let lastSources = [];
            messages.forEach(msg => {
                addMessage('user', msg.question, null, null, '', true, msg.turn_index ?? null);
                const sources = msg.sources_summary || [];
                const diagrams = msg.diagrams || [];
                const exports = msg.exports || [];
                addMessage('bot', msg.answer, sources, null, msg.question, true, msg.turn_index ?? null, diagrams, exports);
                lastSources = sources;
            });
            renderSourcesPanel(lastSources);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    } catch (e) {
        removeLoadingPlaceholder(loading);
        resetConversationUI();
    }
}

async function createNewConversation() {
    const userName = sessionStorage.getItem('userName');
    if (!userName) { document.getElementById('nameModal').style.display = 'flex'; return; }
    if (activeDiagrams.length > 0) {
        if (!confirm(`도식화가 진행 중입니다. 새 대화를 만들면 도식화가 취소됩니다.\n계속하시겠습니까?`)) return;
        [...activeDiagrams].forEach(h => h.cancel());
    }
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
        if (currentLoadingMessage) { removeLoadingPlaceholder(currentLoadingMessage); currentLoadingMessage = null; }
        setButtonState(false);
    }
    try {
        const resp = await fetch('/conversations/new', { method: 'POST' });
        const data = await resp.json();
        const convId = data.conversation_id;
        activeConversationId = convId;
        localStorage.setItem(`activeConversationId_${userName}`, convId);
        await resetServerSession(convId);

        const list = document.getElementById('conv-list');
        const emptyMsg = list.querySelector('.conv-empty, .conv-loading');
        if (emptyMsg) emptyMsg.remove();

        const item = createConvItem(convId, '새 대화', userName, true);
        list.insertBefore(item, list.firstChild);
        document.querySelectorAll('.conv-item').forEach(el => { if (el !== item) el.classList.remove('active'); });

        resetConversationUI();
        questionInput.focus();
    } catch (e) {
        console.warn('새 대화 생성 실패:', e);
    }
}

// ============================================================
// Sources Panel
// ============================================================
function renderSourcesPanel(sources) {
    const body = document.getElementById('sources-panel-body');
    if (!sources || sources.length === 0) {
        body.innerHTML = '<div class="sources-empty">이번 답변에는 참고 이슈가 없습니다.</div>';
        return;
    }
    body.innerHTML = '';
    sources.forEach(s => {
        const card = document.createElement('div');
        card.className = 'source-card';
        if (s.type === 'redmine' || s.issue_id) {
            card.innerHTML = `<div class="source-card-type">Redmine 이슈</div>`;
            const titleDiv = document.createElement('div');
            titleDiv.className = 'source-card-title';
            if (s.url) {
                const a = document.createElement('a');
                a.href = s.url; a.target = '_blank'; a.rel = 'noopener';
                a.textContent = `#${s.issue_id}${s.subject ? ' — ' + s.subject : ''} ↗`;
                a.addEventListener('click', e => { e.preventDefault(); if (s.url) window.open(s.url, '_blank', 'noopener'); });
                titleDiv.appendChild(a);
            } else {
                titleDiv.textContent = `#${s.issue_id}${s.subject ? ' — ' + s.subject : ''}`;
            }
            card.appendChild(titleDiv);
            if (s.project_name) {
                const meta = document.createElement('div');
                meta.className = 'source-card-meta';
                meta.textContent = `프로젝트: ${s.project_name}`;
                card.appendChild(meta);
            }
            const ids = s.attachment_ids || [];
            const fnames = s.attachment_filenames || {};
            const ctypes = s.attachment_content_types || {};
            if (ids.length > 0) {
                const IMAGE_EXTS = ['.png', '.jpg', '.jpeg', '.gif', '.webp'];
                const isImageId = id => {
                    const ct = ctypes[id] || '';
                    const fn = (fnames[id] || '').toLowerCase();
                    return ct.startsWith('image/') || IMAGE_EXTS.some(e => fn.endsWith(e));
                };
                const FILE_ICONS = {
                    'pdf': '📕', 'ppt': '📙', 'pptx': '📙',
                    'doc': '📘', 'docx': '📘', 'xls': '📗', 'xlsx': '📗',
                    'zip': '🗜️', 'tar': '🗜️', 'gz': '🗜️',
                    'hwp': '📄', 'hwpx': '📄', 'txt': '📃', 'md': '📃',
                    'csv': '📊', 'json': '📋', 'py': '🐍',
                    'mp4': '🎬', 'avi': '🎬', 'mov': '🎬',
                };
                const getFileIcon = fname => {
                    const ext = (fname || '').split('.').pop().toLowerCase();
                    return FILE_ICONS[ext] || '📎';
                };

                const imageIds = ids.filter(isImageId);
                const fileIds = ids.filter(id => !isImageId(id));

                // 이미지 인라인 표시
                if (imageIds.length > 0) {
                    const imgsDiv = document.createElement('div');
                    imgsDiv.className = 'source-card-imgs';
                    imageIds.slice(0, 4).forEach(id => {
                        const img = document.createElement('img');
                        img.className = 'source-card-img';
                        img.src = `/redmine-image/${id}`;
                        img.title = fnames[id] || '';
                        img.onerror = () => { img.style.display = 'none'; };
                        img.addEventListener('click', () => {
                            const urls = imageIds.map(i => ({ url: `/redmine-image/${i}`, fname: fnames[i] || '' }));
                            openGallery(`Issue #${s.issue_id}${s.subject ? ' — ' + s.subject : ''}`, urls, imageIds.indexOf(id));
                        });
                        imgsDiv.appendChild(img);
                    });
                    card.appendChild(imgsDiv);
                }

                // 비이미지 파일 다운로드 링크
                if (fileIds.length > 0) {
                    const filesDiv = document.createElement('div');
                    filesDiv.className = 'source-card-files';
                    fileIds.forEach(id => {
                        const fname = fnames[id] || `attachment_${id}`;
                        const a = document.createElement('a');
                        a.className = 'source-card-file-link';
                        a.href = `/redmine-image/${id}`;
                        a.download = fname;  // 다운로드 시 파일명 지정
                        a.textContent = `${getFileIcon(fname)} ${fname}`;
                        a.title = fname;
                        filesDiv.appendChild(a);
                    });
                    card.appendChild(filesDiv);
                }
            }
        } else if (s.type === 'crf' || s.record_id) {
            card.innerHTML = `<div class="source-card-type">CRF 임상 데이터</div>
                <div class="source-card-title">${escapeHtml(s.record_id || '')}</div>
                <div class="source-card-meta">병원: ${escapeHtml(s.hospital || 'N/A')} | 암종: ${escapeHtml(s.cancer_type || s.sheet || 'N/A')}</div>
                ${s.path_no ? `<div class="source-card-meta">병리번호: ${escapeHtml(s.path_no)}</div>` : ''}`;
        } else if (s.type === 'document' || s.filename) {
            card.innerHTML = `<div class="source-card-type">문서</div>
                <div class="source-card-title">${escapeHtml(s.filename || '')}</div>
                <div class="source-card-meta">청크 ${(s.chunk_index ?? 0) + 1} / ${s.total_chunks || '?'}</div>`;
        } else {
            card.innerHTML = `<div class="source-card-type">참고 자료</div><div class="source-card-title">${escapeHtml(JSON.stringify(s))}</div>`;
        }
        body.appendChild(card);
    });
}

function toggleMobileSources() {
    const panel = document.getElementById('sources-panel');
    const overlay = document.getElementById('sources-overlay-bg');
    const isOpen = panel.classList.contains('mobile-sources-open');
    if (isOpen) {
        panel.classList.remove('mobile-sources-open');
        if (overlay) overlay.classList.remove('visible');
    } else {
        panel.classList.add('mobile-sources-open');
        if (overlay) overlay.classList.add('visible');
    }
}

// ============================================================
// 메시지 추가
// ============================================================
function addLoadingPlaceholder() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot loading';
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.style.justifyContent = 'center';
    const dots = document.createElement('div');
    dots.className = 'ellipsis-dots';
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('span');
        dot.className = 'dot';
        dots.appendChild(dot);
    }
    contentDiv.appendChild(dots);
    messageDiv.appendChild(contentDiv);
    messagesInner.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return messageDiv;
}

function removeLoadingPlaceholder(el) { if (el && el.parentNode) el.remove(); }

function addMessage(role, content, sources = null, charts = null, question = '', isHistory = false, turnIndex = null, diagrams = null, exports = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    if (turnIndex !== null) messageDiv.dataset.turnIndex = turnIndex;
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = convertMarkdownTable(content);

    // 인라인 이미지 클릭 → 갤러리 열기 (sources 패널 클릭과 충돌 방지)
    contentDiv.querySelectorAll('.answer-inline-img').forEach(img => {
        img.addEventListener('click', e => {
            e.stopPropagation();
            openGallery('', [{ url: img.getAttribute('src'), fname: img.getAttribute('alt') || '' }], 0);
        });
    });

    if (charts && charts.length > 0) {
        const chartsContainer = document.createElement('div');
        chartsContainer.style.cssText = 'margin:15px 0;';
        charts.forEach((chart, idx) => {
            const img = document.createElement('img');
            img.src = `data:${chart.mime_type};base64,${chart.data}`;
            img.alt = `Chart ${idx + 1}`;
            img.style.cssText = 'max-width:100%;height:auto;border-radius:8px;margin:10px 0;box-shadow:0 2px 8px rgba(0,0,0,0.1);';
            chartsContainer.appendChild(img);
        });
        contentDiv.appendChild(chartsContainer);
    }

    const timestampDiv = document.createElement('div');
    timestampDiv.className = 'timestamp';
    timestampDiv.textContent = isHistory ? '' : new Date().toLocaleTimeString('ko-KR');
    contentDiv.appendChild(timestampDiv);

    if (role === 'bot' && content && content.length > 80) {
        const section = _buildDiagramSection(question, content);
        // 저장된 도식화 이미지 복원
        if (diagrams && diagrams.length > 0) {
            const diagramImg = section.querySelector('.diagram-image');
            const progressLabel = section.querySelector('.diagram-progress');
            const downloadGroup = section.querySelector('.diagram-download-group');
            const downloadFormatInput = section.querySelector('.diagram-download-format-input');
            const downloadBtn = section.querySelector('.diagram-download-btn');
            const last = diagrams[diagrams.length - 1];
            // 파일 경로 기반 로드 (구버전 image_base64 인라인 방식도 지원)
            if (last.file_path) {
                fetch(`/diagram-image?path=${encodeURIComponent(last.file_path)}`)
                    .then(r => r.ok ? r.json() : null)
                    .then(d => {
                        if (d && d.image_base64) {
                            diagramImg.src = d.image_base64;
                            diagramImg.style.display = 'block';
                        }
                    })
                    .catch(() => {});
            } else if (last.image_base64) {
                diagramImg.src = last.image_base64;
                diagramImg.style.display = 'block';
            }
            progressLabel.style.display = 'block';
            progressLabel.textContent = `[${last.mode === 'patent' ? '특허 도식화' : '일반 도식화'}] 생성 완료 (복원됨)`;
            downloadGroup.style.display = '';
            downloadBtn.onclick = () => convertAndDownload(diagramImg, downloadFormatInput ? downloadFormatInput.value : 'png', `diagram_${last.task_id}`);
        }
        contentDiv.appendChild(section);
    }

    // 봇 메시지 하단 액션 바
    if (role === 'bot' && content && content.length > 0) {
        const actionBar = document.createElement('div');
        actionBar.className = 'msg-action-bar';

        // 복사 버튼 (이모티콘만)
        const copyBtn = document.createElement('button');
        copyBtn.className = 'msg-action-btn';
        copyBtn.title = '답변 복사';
        copyBtn.innerHTML = '<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="5" y="1" width="9" height="11" rx="1.5"/><rect x="1" y="4" width="9" height="11" rx="1.5"/></svg>';
        copyBtn.addEventListener('click', e => {
            e.stopPropagation();
            const markSuccess = () => {
                copyBtn.innerHTML = '&#10003;';
                copyBtn.classList.add('copied');
                setTimeout(() => { copyBtn.innerHTML = '<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="5" y="1" width="9" height="11" rx="1.5"/><rect x="1" y="4" width="9" height="11" rx="1.5"/></svg>'; copyBtn.classList.remove('copied'); }, 2000);
            };
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(content).then(markSuccess).catch(() => fallbackCopy(content, markSuccess));
            } else {
                fallbackCopy(content, markSuccess);
            }
        });

        // ... 더보기 메뉴
        const moreWrap = document.createElement('div');
        moreWrap.className = 'msg-more-wrap';

        const moreBtn = document.createElement('button');
        moreBtn.className = 'msg-more-btn';
        moreBtn.title = '추가 기능';
        moreBtn.innerHTML = '<svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor"><circle cx="2.5" cy="8" r="1.4"/><circle cx="8" cy="8" r="1.4"/><circle cx="13.5" cy="8" r="1.4"/></svg>';

        const moreMenu = document.createElement('div');
        moreMenu.className = 'msg-more-menu';

        const menuItems = [
            { icon: '', label: '도식화 생성', action: () => { moreMenu.classList.remove('open'); requestDiagram({dataset:{question, answer:content, mode:'default'}}, _getDiagramSection(contentDiv)); }},
            { icon: '', label: '특허 도식화 생성', action: () => { moreMenu.classList.remove('open'); requestDiagram({dataset:{question, answer:content, mode:'patent'}}, _getDiagramSection(contentDiv)); }},
            { icon: '', label: 'DOCX 내보내기', cls: 'separator', action: () => { moreMenu.classList.remove('open'); _triggerDocxFromTurn(turnIndex, content, question, actionBar); }},
            { icon: '', label: 'MD 내보내기', action: () => { moreMenu.classList.remove('open'); _exportAsMd(content, question); }},
        ];

        menuItems.forEach(item => {
            const btn = document.createElement('button');
            btn.className = 'msg-more-item' + (item.cls ? ' ' + item.cls : '');
            btn.textContent = item.icon ? `${item.icon} ${item.label}` : item.label;
            btn.addEventListener('click', e => { e.stopPropagation(); item.action(); });
            moreMenu.appendChild(btn);
        });

        moreBtn.addEventListener('click', e => {
            e.stopPropagation();
            const isOpen = moreMenu.classList.contains('open');
            document.querySelectorAll('.msg-more-menu.open').forEach(m => m.classList.remove('open'));
            if (!isOpen) moreMenu.classList.add('open');
        });

        moreWrap.appendChild(moreBtn);
        moreWrap.appendChild(moreMenu);
        actionBar.appendChild(copyBtn);
        actionBar.appendChild(moreWrap);
        contentDiv.appendChild(actionBar);
    }

    // 봇 메시지 클릭 시 해당 턴의 Sources 패널로 교체
    if (role === 'bot' && sources !== null) {
        const _sources = sources;
        messageDiv.style.cursor = 'pointer';
        messageDiv.title = '클릭하면 이 답변의 참고 이슈를 표시합니다';
        messageDiv.addEventListener('click', () => renderSourcesPanel(_sources));
    }

    messageDiv.appendChild(contentDiv);
    messagesInner.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return messageDiv;
}

function _getDiagramSection(contentDiv) {
    let sec = contentDiv.querySelector('.diagram-section');
    if (!sec) {
        sec = document.createElement('div');
        sec.className = 'diagram-section';
        sec.style.cssText = 'margin-top:8px;padding-top:8px;border-top:1px solid #eef2f7;';
        const progressLabel = document.createElement('div');
        progressLabel.className = 'diagram-progress';
        const stopBtn = document.createElement('button');
        stopBtn.className = 'diagram-stop-btn';
        stopBtn.innerHTML = '중지';
        const diagramImg = document.createElement('img');
        diagramImg.className = 'diagram-image';
        diagramImg.alt = '생성된 도식';
        const downloadGroup = document.createElement('div');
        downloadGroup.className = 'diagram-download-group';
        const { formatInput, formatSelect, downloadBtn } = _buildDownloadGroup();
        downloadGroup.appendChild(formatInput);
        downloadGroup.appendChild(formatSelect);
        downloadGroup.appendChild(downloadBtn);
        downloadBtn.onclick = () => convertAndDownload(diagramImg, formatInput.value, 'diagram_task');
        sec.appendChild(progressLabel);
        sec.appendChild(stopBtn);
        sec.appendChild(diagramImg);
        sec.appendChild(downloadGroup);
        // actionBar 앞에 삽입
        const actionBar = contentDiv.querySelector('.msg-action-bar');
        if (actionBar) contentDiv.insertBefore(sec, actionBar);
        else contentDiv.appendChild(sec);
    }
    return sec;
}

function _buildDiagramSection(question, content) {
    const diagramSection = document.createElement('div');
    diagramSection.className = 'diagram-section';
    diagramSection.style.cssText = 'margin-top:8px;padding-top:8px;border-top:1px solid #eef2f7;';

    const stopBtn = document.createElement('button');
    stopBtn.className = 'diagram-stop-btn';
    stopBtn.innerHTML = '&#9209; 중지';

    const progressLabel = document.createElement('div');
    progressLabel.className = 'diagram-progress';

    const diagramImg = document.createElement('img');
    diagramImg.className = 'diagram-image';
    diagramImg.alt = '생성된 도식';

    const downloadGroup = document.createElement('div');
    downloadGroup.className = 'diagram-download-group';
    const { formatInput, formatSelect, downloadBtn } = _buildDownloadGroup();
    downloadGroup.appendChild(formatInput);
    downloadGroup.appendChild(formatSelect);
    downloadGroup.appendChild(downloadBtn);
    downloadBtn.onclick = () => convertAndDownload(diagramImg, formatInput.value, `diagram_task`);

    diagramSection.appendChild(progressLabel);
    diagramSection.appendChild(stopBtn);
    diagramSection.appendChild(diagramImg);
    diagramSection.appendChild(downloadGroup);
    return diagramSection;
}

function fallbackCopy(text, cb) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(cb).catch(() => {
            // clipboard API 실패 시 textarea execCommand fallback
            _execCommandCopy(text, cb);
        });
    } else {
        _execCommandCopy(text, cb);
    }
}
function _execCommandCopy(text, cb) {
    try {
        const ta = document.createElement('textarea');
        ta.value = text; ta.style.cssText = 'position:fixed;opacity:0;top:0;left:0;';
        document.body.appendChild(ta); ta.focus(); ta.select();
        // eslint-disable-next-line no-restricted-syntax
        const ok = document.execCommand('copy'); // HTTPS 불가 환경 최후 수단
        document.body.removeChild(ta);
        if (ok) cb();
    } catch (_) {}
}

function _buildDownloadGroup() {
    const formatInput = document.createElement('input');
    formatInput.type = 'hidden';
    formatInput.className = 'diagram-download-format-input';
    formatInput.value = 'png';

    const formatSelect = document.createElement('div');
    formatSelect.className = 'diagram-download-format-select';
    formatSelect.setAttribute('role', 'combobox');
    formatSelect.setAttribute('aria-haspopup', 'listbox');
    formatSelect.setAttribute('aria-expanded', 'false');

    const trigger = document.createElement('button');
    trigger.type = 'button';
    trigger.className = 'diagram-download-format-trigger';
    trigger.setAttribute('aria-expanded', 'false');

    const triggerText = document.createElement('span');
    triggerText.className = 'diagram-download-format-text';
    triggerText.textContent = 'PNG';
    const caret = document.createElement('span');
    caret.className = 'diagram-download-format-caret';
    caret.setAttribute('aria-hidden', 'true');
    caret.textContent = '▾';
    trigger.appendChild(triggerText);
    trigger.appendChild(caret);

    const menu = document.createElement('ul');
    menu.className = 'diagram-download-format-menu';
    menu.setAttribute('role', 'listbox');

    const setFormat = (value, label) => {
        formatInput.value = value;
        triggerText.textContent = label;
        menu.querySelectorAll('li[role="option"]').forEach(opt => {
            opt.setAttribute('aria-selected', opt.dataset.value === value ? 'true' : 'false');
            opt.classList.toggle('is-active', opt.dataset.value === value);
        });
    };
    [{ value: 'png', label: 'PNG' }, { value: 'jpeg', label: 'JPEG' }, { value: 'webp', label: 'WebP' }, { value: 'pdf', label: 'PDF' }].forEach((fmt, idx) => {
        const opt = document.createElement('li');
        opt.setAttribute('role', 'option');
        opt.dataset.value = fmt.value;
        opt.textContent = fmt.label;
        opt.setAttribute('aria-selected', idx === 0 ? 'true' : 'false');
        if (idx === 0) opt.classList.add('is-active');
        menu.appendChild(opt);
    });

    trigger.addEventListener('click', e => {
        e.preventDefault(); e.stopPropagation();
        const isOpen = formatSelect.classList.contains('is-open');
        closeAllDiagramDownloadDropdowns();
        if (!isOpen) {
            formatSelect.classList.add('is-open');
            formatSelect.setAttribute('aria-expanded', 'true');
            trigger.setAttribute('aria-expanded', 'true');
        }
    });
    menu.addEventListener('click', e => {
        const option = e.target.closest('li[role="option"]');
        if (!option) return;
        setFormat(option.dataset.value, option.textContent.trim());
        closeDiagramDownloadDropdown(formatSelect);
    });

    formatSelect.appendChild(trigger);
    formatSelect.appendChild(menu);

    const downloadBtn = document.createElement('button');
    downloadBtn.className = 'diagram-download-btn';
    downloadBtn.innerHTML = '&#11123; 다운로드';

    return { formatInput, formatSelect, downloadBtn };
}

// ============================================================
// 전송 로직
// ============================================================
function setButtonState(isCancel) {
    sendBtn.disabled = false;
    sendBtn.innerHTML = isCancel ? '취소' : '전송';
    sendBtn.className = isCancel ? 'cancel-btn' : '';
    sendBtn.onclick = isCancel ? cancelRequest : sendMessage;
}

function cancelRequest() {
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
        removeLoadingPlaceholder(currentLoadingMessage);
        currentLoadingMessage = null;
        setButtonState(false);
        addMessage('bot', '요청이 취소되었습니다.');
    }
}

async function sendMessage() {
    const question = questionInput.value.trim();
    if (!question) return;
    const userName = sessionStorage.getItem('userName');
    if (!userName) {
        alert('먼저 이름을 입력해주세요');
        document.getElementById('nameModal').style.display = 'flex';
        return;
    }
    if (!activeConversationId) {
        try {
            const resp = await fetch('/conversations/new', { method: 'POST' });
            const data = await resp.json();
            activeConversationId = data.conversation_id;
            localStorage.setItem(`activeConversationId_${userName}`, activeConversationId);
            await resetServerSession(activeConversationId);
            const list = document.getElementById('conv-list');
            const emptyMsg = list.querySelector('.conv-empty, .conv-loading');
            if (emptyMsg) emptyMsg.remove();
            const item = createConvItem(activeConversationId, '새 대화', userName, true);
            list.insertBefore(item, list.firstChild);
            document.querySelectorAll('.conv-item').forEach(el => { if (el !== item) el.classList.remove('active'); });
        } catch (e) {
            addMessage('bot', '대화를 시작할 수 없습니다. 다시 시도해주세요.');
            return;
        }
    }
    const wb = document.getElementById('welcome-msg');
    if (wb) wb.remove();

    addMessage('user', question);
    questionInput.value = '';
    autoResizeTextarea();

    currentAbortController = new AbortController();
    currentLoadingMessage = addLoadingPlaceholder();
    setButtonState(true);

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, user_name: userName, conversation_id: activeConversationId }),
            signal: currentAbortController.signal
        });
        const data = await response.json();
        removeLoadingPlaceholder(currentLoadingMessage);
        currentLoadingMessage = null;
        if (response.ok) {
            // DOCX export 트리거 응답 처리
            if (data.docx_export && data.export_id) {
                removeLoadingPlaceholder(currentLoadingMessage);
                currentLoadingMessage = null;
                // 새 bot 메시지/대화 제목 갱신 없음 — target_turn에 진행 UI 부착
                _startDocxExport(data.export_id, data.target_turn_index);
                return;
            }

            const sources = data.sources || [];
            addMessage('bot', data.answer, sources, data.charts || null, question, false, data.turn_index ?? null);
            renderSourcesPanel(sources);
            const convItem = document.querySelector(`.conv-item[data-conv-id="${activeConversationId}"]`);
            if (convItem && !localStorage.getItem(`convTitle_${activeConversationId}`) &&
                (convItem.dataset.displayTitle === '새 대화' || convItem.dataset.displayTitle === '' || !convItem.dataset.displayTitle)) {
                const title = question.slice(0, 30) + (question.length > 30 ? '...' : '');
                const titleEl = convItem.querySelector('.conv-item-title');
                if (titleEl) titleEl.textContent = title;
                convItem.title = title;
                convItem.dataset.displayTitle = title;
                saveConvTitle(activeConversationId, title);
            }
        } else {
            addMessage('bot', `오류: ${data.error}`);
        }
    } catch (error) {
        removeLoadingPlaceholder(currentLoadingMessage);
        currentLoadingMessage = null;
        if (error.name !== 'AbortError') addMessage('bot', `네트워크 오류: ${error.message}`);
    } finally {
        currentAbortController = null;
        setButtonState(false);
    }
}

function autoResizeTextarea() {
    questionInput.style.height = 'auto';
    questionInput.style.height = Math.min(questionInput.scrollHeight, 160) + 'px';
}
questionInput.addEventListener('input', autoResizeTextarea);
questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        sendMessage();
    }
});
document.getElementById('userNameInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') setUserName();
});

// ============================================================
// PaperBanana 도식화
// ============================================================
async function requestDiagram(btn, section) {
    const userName = sessionStorage.getItem('userName') || '';
    const question = btn.dataset.question || '';
    const ragAnswer = btn.dataset.answer || '';
    const mode = btn.dataset.mode || 'default';

    const progressLabel = section.querySelector('.diagram-progress');
    const diagramImg = section.querySelector('.diagram-image');
    const stopBtn = section.querySelector('.diagram-stop-btn');
    const downloadGroup = section.querySelector('.diagram-download-group');
    const downloadFormatInput = section.querySelector('.diagram-download-format-input');
    const downloadBtn = section.querySelector('.diagram-download-btn');
    const allDiagramBtns = section.querySelectorAll('.diagram-btn');

    const t0 = Date.now();
    let currentEs = null;
    let cancelled = false;
    let abortController = new AbortController();
    let taskId;

    let timerInterval = null;
    function startTimer(prefix) {
        clearInterval(timerInterval);
        timerInterval = setInterval(() => {
            progressLabel.textContent = `${prefix} (${((Date.now() - t0) / 1000).toFixed(0)}s 경과)`;
        }, 1000);
    }
    function stopTimer() { clearInterval(timerInterval); timerInterval = null; }
    function showStopBtn() { stopBtn.style.display = 'inline-flex'; }
    function hideStopBtn() { stopBtn.style.display = 'none'; }

    const originalLabels = new Map();
    if (allDiagramBtns) allDiagramBtns.forEach(b => { originalLabels.set(b, b.innerHTML); });
    let _diagramDone = false;

    // 전역 취소 핸들러 등록
    const diagramHandle = {
        cancel: () => {
            if (_diagramDone) return;
            cancelled = true;
            abortController.abort();
            if (currentEs) { currentEs.close(); currentEs = null; }
            resetBtn();
            progressLabel.textContent = '도식화 생성이 중지되었습니다.';
            if (taskId) fetch(`/pb-cancel/${taskId}`, { method: 'DELETE' }).catch(() => {});
        }
    };
    activeDiagrams.push(diagramHandle);

    function resetBtn() {
        if (!_diagramDone) {
            _diagramDone = true;
            activeDiagramCount = Math.max(0, activeDiagramCount - 1);
            const idx = activeDiagrams.indexOf(diagramHandle);
            if (idx !== -1) activeDiagrams.splice(idx, 1);
        }
        stopTimer(); hideStopBtn();
        if (allDiagramBtns) allDiagramBtns.forEach(b => { b.innerHTML = originalLabels.get(b); b.disabled = false; });
    }

    stopBtn.onclick = function() { diagramHandle.cancel(); };

    activeDiagramCount++;
    if (allDiagramBtns) allDiagramBtns.forEach(b => { b.disabled = true; });
    if (btn.innerHTML !== undefined && btn.dataset) btn.innerHTML = '생성 중...';
    downloadGroup.style.display = 'none';
    progressLabel.style.display = 'block';
    progressLabel.textContent = 'LLM 재작성 중...';
    showStopBtn();
    startTimer('LLM 재작성 중');

    let rewrittenData;
    try {
        const resp = await fetch('/visualize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, rag_answer: ragAnswer, user_name: userName, mode }),
            signal: abortController.signal,
        });
        if (cancelled) return;
        const data = await resp.json();
        if (!resp.ok) {
            stopTimer();
            progressLabel.textContent = resp.status === 429 ? '요청이 많습니다. 잠시 후 다시 시도하세요.' : `오류: ${data.error || resp.status}`;
            resetBtn(); return;
        }
        if (cancelled) return;
        rewrittenData = data;
    } catch (err) {
        if (cancelled) return;
        stopTimer(); progressLabel.textContent = `네트워크 오류: ${err.message}`; resetBtn(); return;
    }

    startTimer('paperbanana 시작 중');
    try {
        const pbResp = await fetch('/pb-start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source_context: rewrittenData.source_context, communicative_intent: rewrittenData.communicative_intent }),
            signal: abortController.signal,
        });
        if (cancelled) return;
        const pbData = await pbResp.json();
        if (!pbResp.ok) { stopTimer(); progressLabel.textContent = `오류: ${pbData.error || pbResp.status}`; resetBtn(); return; }
        taskId = pbData.task_id;
        if (cancelled) { if (taskId) fetch(`/pb-cancel/${taskId}`, { method: 'DELETE' }).catch(() => {}); return; }
    } catch (err) {
        if (cancelled) return;
        stopTimer(); progressLabel.textContent = `네트워크 오류: ${err.message}`; resetBtn(); return;
    }

    if (cancelled) return;
    startTimer('paperbanana 서버 연결 중');
    const agentLabels = { Retriever: '참고 예시 검색 중', Planner: '도식 구조 설계 중', Stylist: '스타일 정제 중', Visualizer: '이미지 생성 중', Critic: '품질 검토 중', Complete: '완료' };

    await new Promise((resolve) => {
        const es = new EventSource(`/pb-stream/${taskId}`);
        currentEs = es;
        es.onmessage = async (e) => {
            if (cancelled) { es.close(); resolve(); return; }
            let event;
            try { event = JSON.parse(e.data); } catch { return; }
            const elapsed = ((Date.now() - t0) / 1000).toFixed(0);
            if (event.error && !event.done) {
                stopTimer(); progressLabel.textContent = `오류: ${event.error}`;
                resetBtn(); es.close(); currentEs = null; resolve(); return;
            }
            if (event.queued) { progressLabel.textContent = `⏳ 대기 중 (${event.queue_position}번째)...`; return; }
            if (!event.done && event.agent !== 'Complete') {
                stopTimer();
                progressLabel.textContent = `[${event.progress || 0}%] ${agentLabels[event.agent] || event.agent} (${elapsed}s 경과)`;
                return;
            }
            es.close(); currentEs = null; stopTimer(); hideStopBtn();
            const totalElapsed = event.elapsed_seconds || ((Date.now() - t0) / 1000).toFixed(1);
            progressLabel.textContent = `이미지 다운로드 중... (${totalElapsed}s)`;
            try {
                const imgResp = await fetch(`/pb-image/${taskId}`);
                const imgData = await imgResp.json();
                if (imgResp.ok && imgData.image_base64) {
                    diagramImg.src = imgData.image_base64;
                    diagramImg.style.display = 'block';
                    progressLabel.textContent = `[${mode === 'patent' ? '특허 도식화' : '일반 도식화'}] 생성 완료 (${totalElapsed}s)`;
                    resetBtn();
                    downloadBtn.onclick = () => convertAndDownload(diagramImg, downloadFormatInput ? downloadFormatInput.value : 'png', `diagram_${taskId}`);
                    downloadGroup.style.display = 'inline-flex';
                    // 도식화 이미지를 DB에 저장 (대화 전환 후 복원용)
                    const msgEl = section.closest('.message');
                    const turnIdx = msgEl ? parseInt(msgEl.dataset.turnIndex ?? '-1') : -1;
                    const uName = sessionStorage.getItem('userName') || '';
                    if (uName && activeConversationId && turnIdx >= 0) {
                        fetch('/save-diagram', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                user_name: uName,
                                conversation_id: activeConversationId,
                                turn_index: turnIdx,
                                image_base64: imgData.image_base64,
                                task_id: taskId,
                                mode: mode
                            })
                        }).catch(() => {});
                    }
                } else {
                    progressLabel.textContent = `이미지 수신 실패: ${imgData.error || ''}`; resetBtn();
                }
            } catch (err) {
                progressLabel.textContent = `이미지 수신 오류: ${err.message}`; resetBtn();
            }
            chatBox.scrollTop = chatBox.scrollHeight;
            resolve();
        };
        es.onerror = () => {
            es.close(); currentEs = null; stopTimer();
            if (!cancelled) { progressLabel.textContent = 'SSE 연결 오류. 잠시 후 다시 시도하세요.'; resetBtn(); }
            resolve();
        };
    });
    chatBox.scrollTop = chatBox.scrollHeight;
}

// ============================================================
// 갤러리 / 이미지 뷰어
// ============================================================
let _galleryUrls = [], _galleryIdx = 0;
function openGallery(title, urls, startIdx = 0) {
    _galleryUrls = urls; _galleryIdx = startIdx;
    document.getElementById('imgGalleryTitle').textContent = title;
    updateGallery();
    document.getElementById('imgGalleryModal').classList.add('active');
}
function updateGallery() {
    const item = _galleryUrls[_galleryIdx];
    document.getElementById('imgGalleryImg').src = item.url;
    document.getElementById('imgGalleryCounter').textContent = `${_galleryIdx + 1} / ${_galleryUrls.length}${item.fname ? '  ·  ' + item.fname : ''}`;
    document.getElementById('imgGalleryPrev').disabled = _galleryIdx === 0;
    document.getElementById('imgGalleryNext').disabled = _galleryIdx === _galleryUrls.length - 1;
}
document.getElementById('imgGalleryPrev').onclick = () => { if (_galleryIdx > 0) { _galleryIdx--; updateGallery(); } };
document.getElementById('imgGalleryNext').onclick = () => { if (_galleryIdx < _galleryUrls.length - 1) { _galleryIdx++; updateGallery(); } };
document.getElementById('imgGalleryClose').onclick = () => document.getElementById('imgGalleryModal').classList.remove('active');
document.getElementById('imgGalleryModal').onclick = (e) => { if (e.target === e.currentTarget) document.getElementById('imgGalleryModal').classList.remove('active'); };
document.addEventListener('keydown', (e) => {
    if (!document.getElementById('imgGalleryModal').classList.contains('active')) return;
    if (e.key === 'ArrowLeft') document.getElementById('imgGalleryPrev').click();
    if (e.key === 'ArrowRight') document.getElementById('imgGalleryNext').click();
    if (e.key === 'Escape') document.getElementById('imgGalleryModal').classList.remove('active');
});

function openImageViewer(url) {
    document.getElementById('imageViewerImg').src = url;
    document.getElementById('imageViewerModal').classList.add('active');
}
document.getElementById('imageViewerClose').onclick = () => document.getElementById('imageViewerModal').classList.remove('active');
document.getElementById('imageViewerModal').onclick = (e) => { if (e.target === e.currentTarget) document.getElementById('imageViewerModal').classList.remove('active'); };

// ============================================================
// 모달 관리
// ============================================================
function showGuideRedmine() { document.getElementById('guideModalRedmine').style.display = 'block'; }
function closeGuideRedmine() { document.getElementById('guideModalRedmine').style.display = 'none'; }
function showGuideCRF() { document.getElementById('guideModalCRF').style.display = 'block'; }
function closeGuideCRF() { document.getElementById('guideModalCRF').style.display = 'none'; }
function showGuideRedmineSite() { document.getElementById('guideModalRedmineSite').style.display = 'block'; }
function closeGuideRedmineSite() { document.getElementById('guideModalRedmineSite').style.display = 'none'; }

async function showUserList() {
    const modal = document.getElementById('userListModal');
    const contentDiv = document.getElementById('userListContent');
    modal.style.display = 'block';
    contentDiv.innerHTML = '<div class="modal-status">로딩 중...</div>';
    try {
        const response = await fetch('/users');
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || '사용자 목록을 가져올 수 없습니다');
        const users = data.users || [];
        if (users.length === 0) { contentDiv.innerHTML = '<div class="modal-empty">등록된 사용자가 없습니다.</div>'; return; }
        const rows = users.map(user => {
            const firstSeen = user.first_seen ? new Date(user.first_seen).toLocaleString('ko-KR') : '-';
            const lastSeen = user.last_seen ? new Date(user.last_seen).toLocaleString('ko-KR') : '-';
            const safeName = escapeHtml(user.user_name);
            return `<tr>
                <td><strong>${safeName}</strong></td>
                <td class="center">${user.total_turns ?? user.total_conversations ?? 0} 턴</td>
                <td class="center">${escapeHtml(firstSeen)}</td>
                <td class="center">${escapeHtml(lastSeen)}</td>
                <td class="center"><button type="button" class="delete-user-btn" data-user="${safeName}">삭제</button></td>
            </tr>`;
        }).join('');
        contentDiv.innerHTML = `<div class="user-table-wrap"><table class="user-table">
            <thead><tr><th>사용자명</th><th class="center">질문 수 (턴)</th><th class="center">최초 접속</th><th class="center">최근 접속</th><th class="center">삭제</th></tr></thead>
            <tbody>${rows}</tbody></table></div>
            <div class="user-list-summary">총 <strong>${users.length}</strong>명의 사용자가 등록되어 있습니다.</div>`;
        contentDiv.querySelectorAll('.delete-user-btn').forEach(btn => {
            btn.addEventListener('click', () => deleteUser(btn.dataset.user || ''));
        });
    } catch (error) {
        contentDiv.innerHTML = `<div class="modal-error">오류: ${escapeHtml(error.message)}</div>`;
    }
}
function closeUserList() { document.getElementById('userListModal').style.display = 'none'; }

async function deleteUser(userName) {
    if (!confirm(`정말로 '${userName}' 사용자를 삭제하시겠습니까?\n해당 사용자의 모든 대화 기록이 삭제됩니다.`)) return;
    try {
        const response = await fetch(`/users/${userName}`, { method: 'DELETE' });
        const data = await response.json();
        if (response.ok) { alert(`'${userName}' 사용자가 삭제되었습니다.`); showUserList(); }
        else throw new Error(data.error || '삭제 실패');
    } catch (error) { alert(`삭제 중 오류 발생: ${error.message}`); }
}

window.onclick = (e) => {
    if (e.target.id === 'guideModalRedmine') closeGuideRedmine();
    if (e.target.id === 'guideModalCRF') closeGuideCRF();
    if (e.target.id === 'guideModalRedmineSite') closeGuideRedmineSite();
    if (e.target.id === 'userListModal') closeUserList();
};

function usePrompt(promptText) {
    questionInput.value = promptText;
    autoResizeTextarea();
    closeGuideRedmine(); closeGuideCRF();
    questionInput.focus();
}

// ============================================================
// 페이지 로드 부트스트랩
// ============================================================
window.onload = async () => {
    const userName = sessionStorage.getItem('userName');
    if (!userName) {
        document.getElementById('nameModal').style.display = 'flex';
        document.getElementById('userNameInput').focus();
        return;
    }
    // 탭 복원 시 서버 세션에 user_name 재등록 (이미지 프록시 인증용)
    await initServerSession(userName);
    updateSidebarUser(userName);
    await loadConversationList(userName);

    const deletedIds = getDeletedConvIds();
    const savedConvId = localStorage.getItem(`activeConversationId_${userName}`);
    const convItems = document.querySelectorAll('.conv-item');
    // 삭제된 대화 ID는 복원 대상에서 제외
    const validSavedId = savedConvId && !deletedIds.includes(savedConvId) ? savedConvId : null;
    if (validSavedId && convItems.length > 0) {
        const found = Array.from(convItems).find(el => el.dataset.convId === validSavedId);
        if (found) {
            await selectConversation(validSavedId, userName);
        } else if (convItems.length > 0) {
            await selectConversation(convItems[0].dataset.convId, userName);
        }
    } else if (convItems.length > 0) {
        await selectConversation(convItems[0].dataset.convId, userName);
    }
    questionInput.focus();
    // 새로고침 후 진행 중 DOCX export 복원
    await _restorePendingExports();
};

// ============================================================
// convertAndDownload (jsPDF 의존 — jspdf CDN 로드 후 사용)
// ============================================================
function convertAndDownload(img, format, baseName) {
    const doConvert = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext('2d');
        if (format === 'jpeg') { ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, canvas.width, canvas.height); }
        ctx.drawImage(img, 0, 0);
        if (format === 'pdf') {
            const imgData = canvas.toDataURL('image/png');
            const { jsPDF } = window.jspdf;
            const orientation = canvas.width > canvas.height ? 'l' : 'p';
            const pdf = new jsPDF({ orientation, unit: 'px', format: [canvas.width, canvas.height] });
            pdf.addImage(imgData, 'PNG', 0, 0, canvas.width, canvas.height);
            pdf.save(`${baseName}.pdf`); return;
        }
        const mimeType = format === 'jpeg' ? 'image/jpeg' : `image/${format}`;
        canvas.toBlob((blob) => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url; a.download = `${baseName}.${format}`; a.click();
            URL.revokeObjectURL(url);
        }, mimeType, 0.95);
    };
    if (img.complete && img.naturalWidth > 0) { doConvert(); } else { img.onload = doConvert; }
}

// ============================================================
// DOCX 내보내기 — SSE 처리, 다운로드 링크, 히스토리 복원
// ============================================================

function _exportAsMd(content, question) {
    const fname = (question || 'answer').slice(0, 30).replace(/[\\/:*?"<>|]/g, '_') + '.md';
    const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = fname; a.click();
    URL.revokeObjectURL(url);
}

function _getDocxUserName() {
    return sessionStorage.getItem('userName') || '';
}

// 버튼 클릭으로 특정 turn을 직접 DOCX export
async function _triggerDocxFromTurn(turnIndex, answer, question, actionBar) {
    try {
        const r = await fetch('/docx-direct', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_name: _getDocxUserName(),
                conversation_id: activeConversationId,
                turn_index: turnIndex,
                answer,
                question,
            }),
        });
        const data = await r.json();
        if (data.export_id) {
            // 버튼을 진행 표시로 교체
            const progressDiv = document.createElement('div');
            progressDiv.className = 'msg-action-btn';
            progressDiv.style.color = '#64748b';
            progressDiv.textContent = '[DOCX] 변환 중...';
            actionBar.appendChild(progressDiv);
            _connectDocxStream(data.export_id, actionBar.parentElement, progressDiv);
        }
    } catch (err) {
        console.error('DOCX export 실패:', err);
    }
}

function _appendDocxLink(container, exportId) {
    const a = document.createElement('a');
    a.href = `/download-export/${exportId}?user_name=${encodeURIComponent(_getDocxUserName())}`;
    a.download = '';
    a.textContent = '📄 DOCX 다운로드';
    a.className = 'docx-download-link';
    a.style.cssText = 'display:inline-block;margin-top:8px;padding:4px 10px;border-radius:6px;background:#e8f0fe;color:#1a56db;font-size:13px;text-decoration:none;border:1px solid #c7d7f9;';
    container.appendChild(a);
}

function _appendDocxProgress(container, label) {
    const div = document.createElement('div');
    div.className = 'docx-progress';
    div.style.cssText = 'margin-top:8px;font-size:13px;color:#555;';
    div.textContent = label;
    container.appendChild(div);
    return div;
}

function _findTurnMessageDiv(turnIndex) {
    return document.querySelector(`.message.bot[data-turn-index="${turnIndex}"]`);
}

function _startDocxExport(exportId, targetTurnIndex) {
    // target_turn_index 턴의 bot 메시지 contentDiv에 진행 UI 부착
    const targetDiv = _findTurnMessageDiv(targetTurnIndex);
    const container = targetDiv ? targetDiv.querySelector('.message-content') : null;

    let progressDiv = null;
    if (container) {
        progressDiv = _appendDocxProgress(container, '[DOCX] 생성 준비 중...');
    }

    // POST /docx-start
    fetch('/docx-start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ export_id: exportId, user_name: _getDocxUserName() }),
    }).then(r => r.json()).then(() => {
        // SSE 연결
        _connectDocxStream(exportId, container, progressDiv);
    }).catch(err => {
        if (progressDiv) progressDiv.textContent = `DOCX 생성 실패: ${err.message}`;
    });
}

function _connectDocxStream(exportId, container, progressDiv) {
    const sse = new EventSource(`/docx-stream/${exportId}?user_name=${encodeURIComponent(_getDocxUserName())}`);

    sse.addEventListener('progress', e => {
        const d = JSON.parse(e.data);
        if (progressDiv) progressDiv.textContent = d.message || '';
    });

    sse.addEventListener('complete', e => {
        sse.close();
        const d = JSON.parse(e.data);
        if (progressDiv) progressDiv.remove();
        // 다운로드 링크 UI 없이 바로 다운로드
        const a = document.createElement('a');
        a.href = `${d.docx_url}?user_name=${encodeURIComponent(_getDocxUserName())}`;
        a.download = '';
        a.click();
    });

    sse.addEventListener('error', e => {
        sse.close();
        let msg = 'DOCX 생성 실패';
        try { msg = JSON.parse(e.data).message || msg; } catch (_) {}
        if (progressDiv) progressDiv.textContent = msg;
    });

    sse.onerror = () => {
        sse.close();
        if (progressDiv) progressDiv.textContent = '연결 오류, 새로고침 후 확인하세요.';
    };
}

// 새로고침 시 진행중 export 복원
async function _restorePendingExports() {
    try {
        const r = await fetch(`/exports/pending?user_name=${encodeURIComponent(_getDocxUserName())}`);
        if (!r.ok) return;
        const list = await r.json();
        for (const item of list) {
            const targetDiv = _findTurnMessageDiv(item.turn_index);
            const container = targetDiv ? targetDiv.querySelector('.message-content') : null;
            if (!container) continue;
            const progressDiv = _appendDocxProgress(container, '[DOCX] 생성 진행 중... (재연결)');
            _connectDocxStream(item.export_id, container, progressDiv);
        }
    } catch (_) {}
}
