const header = document.querySelector("[data-header]");
const menuButton = document.querySelector("[data-menu-button]");
const navigation = document.querySelector("[data-navigation]");
const navLinks = [...document.querySelectorAll(".nav-link")];
const revealItems = [...document.querySelectorAll(".reveal")];
const stageButtons = [...document.querySelectorAll("[data-stage]")];
const architectureScroller = document.querySelector("[data-architecture-scroller]");
const figureDialog = document.querySelector("[data-figure-dialog]");
const openFigureButton = document.querySelector("[data-open-figure]");
const closeFigureButton = document.querySelector("[data-close-figure]");
const paperButtons = [...document.querySelectorAll("[data-paper-button]")];
const toast = document.querySelector("[data-toast]");

let toastTimeout;

const closeMenu = () => {
  if (!menuButton || !navigation) return;
  menuButton.setAttribute("aria-expanded", "false");
  navigation.classList.remove("is-open");
  document.body.classList.remove("menu-open");
};

menuButton?.addEventListener("click", () => {
  const isOpen = menuButton.getAttribute("aria-expanded") === "true";
  menuButton.setAttribute("aria-expanded", String(!isOpen));
  navigation?.classList.toggle("is-open", !isOpen);
  document.body.classList.toggle("menu-open", !isOpen);
});

navLinks.forEach((link) => {
  link.addEventListener("click", closeMenu);
});

const updateHeader = () => {
  header?.classList.toggle("is-scrolled", window.scrollY > 12);
};

updateHeader();
window.addEventListener("scroll", updateHeader, { passive: true });

if ("IntersectionObserver" in window) {
  const revealObserver = new IntersectionObserver(
    (entries, observer) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    },
    { rootMargin: "0px 0px -8%", threshold: 0.08 },
  );

  revealItems.forEach((item) => revealObserver.observe(item));

  const sectionObserver = new IntersectionObserver(
    (entries) => {
      const activeEntry = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

      if (!activeEntry) return;
      navLinks.forEach((link) => {
        link.classList.toggle(
          "is-active",
          link.getAttribute("href") === `#${activeEntry.target.id}`,
        );
      });
    },
    { rootMargin: "-28% 0px -58%", threshold: [0, 0.1, 0.25] },
  );

  ["overview", "method", "results", "code"].forEach((id) => {
    const section = document.getElementById(id);
    if (section) sectionObserver.observe(section);
  });
} else {
  revealItems.forEach((item) => item.classList.add("is-visible"));
}

stageButtons.forEach((button) => {
  button.addEventListener("click", () => {
    stageButtons.forEach((stageButton) => stageButton.classList.remove("is-active"));
    button.classList.add("is-active");

    if (!architectureScroller) return;
    const stagePosition = Number(button.dataset.stage || 0);
    const maximumScroll = architectureScroller.scrollWidth - architectureScroller.clientWidth;
    architectureScroller.scrollTo({
      left: Math.max(0, maximumScroll * stagePosition),
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "auto"
        : "smooth",
    });
  });
});

openFigureButton?.addEventListener("click", () => {
  if (!figureDialog) return;
  figureDialog.showModal();
});

closeFigureButton?.addEventListener("click", () => figureDialog?.close());

figureDialog?.addEventListener("click", (event) => {
  if (event.target === figureDialog) figureDialog.close();
});

paperButtons.forEach((button) => {
  button.addEventListener("click", () => {
    if (!toast) return;
    window.clearTimeout(toastTimeout);
    toast.classList.add("is-visible");
    toastTimeout = window.setTimeout(() => toast.classList.remove("is-visible"), 3200);
  });
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  closeMenu();
  toast?.classList.remove("is-visible");
});

window.addEventListener("resize", () => {
  if (window.innerWidth > 780) closeMenu();
});
