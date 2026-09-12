const header = document.querySelector("[data-header]");
const menuButton = document.querySelector("[data-menu-button]");
const navigation = document.querySelector("[data-navigation]");
const navLinks = [...document.querySelectorAll(".nav-link")];
const revealItems = [...document.querySelectorAll(".reveal")];
const figureDialog = document.querySelector("[data-figure-dialog]");
const openFigureButton = document.querySelector("[data-open-figure]");
const closeFigureButton = document.querySelector("[data-close-figure]");
const paperButtons = [...document.querySelectorAll("[data-paper-button]")];
const copyTitleButton = document.querySelector("[data-copy-title]");
const copyLabel = document.querySelector("[data-copy-label]");
const paperTitleText = document.querySelector("[data-paper-title-text]");
const toast = document.querySelector("[data-toast]");

let toastTimeout;
let copyTimeout;

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

navLinks.forEach((link) => link.addEventListener("click", closeMenu));

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
    { rootMargin: "0px 0px -7%", threshold: 0.07 },
  );

  revealItems.forEach((item) => revealObserver.observe(item));

  const navSections = [
    { id: "overview", href: "#overview" },
    { id: "abstract", href: "#overview" },
    { id: "method", href: "#method" },
    { id: "results", href: "#results" },
    { id: "code", href: "#code" },
    { id: "citation", href: "#citation" },
  ];

  const sectionObserver = new IntersectionObserver(
    (entries) => {
      const activeEntry = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

      if (!activeEntry) return;
      const sectionConfig = navSections.find((item) => item.id === activeEntry.target.id);
      navLinks.forEach((link) => {
        link.classList.toggle("is-active", link.getAttribute("href") === sectionConfig?.href);
      });
    },
    { rootMargin: "-28% 0px -58%", threshold: [0, 0.08, 0.2] },
  );

  navSections.forEach((item) => {
    const section = document.getElementById(item.id);
    if (section) sectionObserver.observe(section);
  });
} else {
  revealItems.forEach((item) => item.classList.add("is-visible"));
}

openFigureButton?.addEventListener("click", () => figureDialog?.showModal());
closeFigureButton?.addEventListener("click", () => figureDialog?.close());

figureDialog?.addEventListener("click", (event) => {
  if (event.target === figureDialog) figureDialog.close();
});

const showToast = (message) => {
  if (!toast) return;
  window.clearTimeout(toastTimeout);
  toast.textContent = message;
  toast.classList.add("is-visible");
  toastTimeout = window.setTimeout(() => toast.classList.remove("is-visible"), 3200);
};

paperButtons.forEach((button) => {
  button.addEventListener("click", () => {
    showToast("The paper link will be added after release.");
  });
});

copyTitleButton?.addEventListener("click", async () => {
  const title = paperTitleText?.textContent?.trim();
  if (!title) return;

  try {
    await navigator.clipboard.writeText(title);
    if (copyLabel) copyLabel.textContent = "Copied";
    showToast("Paper title copied to the clipboard.");
    window.clearTimeout(copyTimeout);
    copyTimeout = window.setTimeout(() => {
      if (copyLabel) copyLabel.textContent = "Copy title";
    }, 2200);
  } catch {
    showToast("Select the paper title to copy it.");
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  closeMenu();
  toast?.classList.remove("is-visible");
});

window.addEventListener("resize", () => {
  if (window.innerWidth > 820) closeMenu();
});
