(function () {
  const API_URL = "http://localhost:8000/api/events/track";

  function sendEvent(eventType, metadata = {}) {
    const payload = {
      user_id: localStorage.getItem("user_id") || generateId(),
      session_id: sessionStorage.getItem("session_id") || startSession(),
      event_type: eventType,
      page_url: window.location.href,
      timestamp: new Date().toISOString(),
      metadata
    };

    navigator.sendBeacon(API_URL, JSON.stringify(payload));
  }

  function generateId() {
    const id = "user-" + Math.random().toString(36).substring(2);
    localStorage.setItem("user_id", id);
    return id;
  }

  function startSession() {
    const id = "session-" + Math.random().toString(36).substring(2);
    sessionStorage.setItem("session_id", id);
    return id;
  }

  // page view event
  window.addEventListener("load", () => {
    sendEvent("page_view");
  });

  // click tracking
  document.addEventListener("click", (e) => {
    sendEvent("click", {
      tag: e.target.tagName,
      id: e.target.id,
      classes: e.target.className
    });
  });

  // export globally
  window.tracker = { sendEvent };
})();
