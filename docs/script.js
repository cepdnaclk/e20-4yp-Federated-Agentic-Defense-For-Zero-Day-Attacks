const navLinks = Array.from(document.querySelectorAll('.site-nav a'));
const sections = navLinks
  .map((link) => document.querySelector(link.getAttribute('href')))
  .filter(Boolean);
const topButton = document.getElementById('to-top');

const setActiveSection = () => {
  const headerOffset = 140;
  const scrollPosition = window.scrollY + headerOffset;

  let currentId = sections[0]?.id || '';

  for (const section of sections) {
    if (section.offsetTop <= scrollPosition) {
      currentId = section.id;
    }
  }

  for (const link of navLinks) {
    const targetId = link.getAttribute('href').replace('#', '');
    link.classList.toggle('active', targetId === currentId);
  }
};

window.addEventListener('scroll', () => {
  setActiveSection();
  const shouldShowTopButton = window.scrollY > 450;
  topButton.classList.toggle('show', shouldShowTopButton);
});

for (const link of navLinks) {
  link.addEventListener('click', (event) => {
    event.preventDefault();
    const target = document.querySelector(link.getAttribute('href'));
    if (!target) {
      return;
    }

    const top = target.getBoundingClientRect().top + window.scrollY - 110;
    window.scrollTo({ top, behavior: 'smooth' });
  });
}

topButton.addEventListener('click', () => {
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

setActiveSection();
