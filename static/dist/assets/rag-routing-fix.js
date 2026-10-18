(function () {
  "use strict";

  if (typeof window.fetch !== "function") return;

  const originalFetch = window.fetch.bind(window);
  let activeIndexId = null;

  function requestUrl(input) {
    return typeof input === "string" ? input : input && input.url;
  }

  function matchesPath(url, path) {
    if (!url) return false;
    try {
      return new URL(url, window.location.origin).pathname === path;
    } catch (_) {
      return false;
    }
  }

  window.fetch = async function (input, init) {
    const url = requestUrl(input);

    if (activeIndexId && matchesPath(url, "/api/chat") && init && init.body) {
      try {
        const chatPayload = JSON.parse(init.body);
        const ragPayload = {
          messages: chatPayload.messages,
          index_id: activeIndexId,
          user_id: Number(chatPayload.user_id),
        };
        const ragUrl = new URL("/chat-rag", new URL(url, window.location.origin));
        return originalFetch(ragUrl.toString(), {
          ...init,
          body: JSON.stringify(ragPayload),
        });
      } catch (error) {
        console.error("Unable to prepare document question request:", error);
      }
    }

    const response = await originalFetch(input, init);
    if (response.ok && matchesPath(url, "/api/upload")) {
      try {
        const uploadResult = await response.clone().json();
        if (uploadResult.status === "success" && uploadResult.index_id) {
          activeIndexId = uploadResult.index_id;
        }
      } catch (error) {
        console.error("Unable to read document upload response:", error);
      }
    }
    return response;
  };

  document.addEventListener(
    "click",
    function (event) {
      const target = event.target;
      const removeButton = target && target.closest && target.closest("button.delete-btn");
      if (removeButton && removeButton.parentElement.querySelector("svg[data-type]")) {
        activeIndexId = null;
      }
    },
    true
  );
})();
