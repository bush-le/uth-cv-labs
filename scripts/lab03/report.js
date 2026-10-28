const viewer = document.querySelector('#viewer');
document.querySelectorAll('.image-card img').forEach(image => {
  image.tabIndex = 0;
  const open = () => {
    viewer.querySelector('img').src = image.src;
    viewer.querySelector('img').alt = image.alt;
    viewer.showModal();
  };
  image.addEventListener('click', open);
  image.addEventListener('keydown', event => { if (event.key === 'Enter') open(); });
});
viewer.querySelector('button').addEventListener('click', () => viewer.close());
viewer.addEventListener('click', event => { if (event.target === viewer) viewer.close(); });
if ('IntersectionObserver' in window) {
  const links = [...document.querySelectorAll('nav a')];
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) links.forEach(link => {
        if (link.hash === '#' + entry.target.id) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    });
  }, { rootMargin: '-10% 0px -65% 0px' });
  document.querySelectorAll('.stage, #flow').forEach(element => observer.observe(element));
}
window.addEventListener('beforeprint', () => document.querySelectorAll('details').forEach(detail => {
  detail.dataset.previous = detail.open;
  detail.open = true;
}));
window.addEventListener('afterprint', () => document.querySelectorAll('details').forEach(detail => {
  detail.open = detail.dataset.previous === 'true';
}));
