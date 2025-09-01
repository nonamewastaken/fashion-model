// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Add loading animation to all elements
    const elements = document.querySelectorAll('.navbar, .hero, .cards-section');
    elements.forEach(element => {
        element.classList.add('loading');
    });

    // Client-side routing
    const routes = {
        'home': '/',
        'fashion-decision-predictor': '/fashion-decision-predictor',
        'trend-insights': '/trend-insights',
        'data-verifier': '/data-verifier',
        'revenue-insights': '/revenue-insights',
        'user-guide': '/user-guide'
    };

    // Handle navigation clicks
    function handleNavigation(route) {
        // Update active state
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        
        // Find and activate the clicked link
        const activeLink = document.querySelector(`[data-route="${route}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }

        // Navigate to the route
        if (routes[route]) {
            window.location.href = routes[route];
        }
    }

    // Add click handlers for navigation links
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const route = this.getAttribute('data-route');
            if (route) {
                handleNavigation(route);
            }
        });
    });

    // Handle dropdown toggle
    const dropdownToggle = document.querySelector('.dropdown-toggle');
    const dropdown = document.querySelector('.dropdown');
    
    if (dropdownToggle && dropdown) {
        dropdownToggle.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            dropdown.classList.toggle('show');
            
            // Add/remove body class for additional CSS control
            if (dropdown.classList.contains('show')) {
                document.body.classList.add('dropdown-open');
            } else {
                document.body.classList.remove('dropdown-open');
            }
        });
        
        // Close dropdown when clicking outside
        document.addEventListener('click', function(e) {
            if (!dropdown.contains(e.target)) {
                dropdown.classList.remove('show');
                document.body.classList.remove('dropdown-open');
            }
        });
        
        // Handle Help icon hover effects when dropdown is open
        dropdownToggle.addEventListener('mouseenter', function() {
            if (dropdown.classList.contains('show')) {
                this.style.background = '#e9ecef';
                this.style.color = '#6366f0';
                this.style.transform = 'scale(1.05)';
            }
        });
        
        dropdownToggle.addEventListener('mouseleave', function() {
            if (dropdown.classList.contains('show')) {
                this.style.background = 'none';
                this.style.color = '#6366f0';
                this.style.transform = 'none';
            }
        });
        
        // Disable other icons when dropdown is open
        dropdown.addEventListener('click', function() {
            const otherIcons = document.querySelectorAll('.nav-icons .tooltip:not(.dropdown .tooltip) .icon-btn');
            const otherTooltips = document.querySelectorAll('.nav-icons .tooltip:not(.dropdown .tooltip) .tooltiptext');
            
            if (this.classList.contains('show')) {
                otherIcons.forEach(icon => {
                    icon.style.pointerEvents = 'none';
                    icon.style.cursor = 'default';
                    icon.style.background = 'none';
                    icon.style.color = '#6c757d';
                    icon.style.transform = 'none';
                });
                otherTooltips.forEach(tooltip => {
                    tooltip.style.display = 'none';
                    tooltip.style.visibility = 'hidden';
                    tooltip.style.opacity = '0';
                });
            } else {
                otherIcons.forEach(icon => {
                    icon.style.pointerEvents = 'auto';
                    icon.style.cursor = 'pointer';
                    icon.style.background = '';
                    icon.style.color = '';
                    icon.style.transform = '';
                });
                otherTooltips.forEach(tooltip => {
                    tooltip.style.display = '';
                    tooltip.style.visibility = '';
                    tooltip.style.opacity = '';
                });
            }
        });
    }

    // Add click handlers for dropdown links
    const dropdownLinks = document.querySelectorAll('.dropdown-content a');
    dropdownLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const route = this.getAttribute('data-route');
            if (route) {
                handleNavigation(route);
            }
            // Close dropdown after navigation
            dropdown.classList.remove('show');
        });
    });

    // Add click handlers for cards
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        card.addEventListener('click', function(e) {
            e.preventDefault();
            const route = this.getAttribute('data-route');
            if (route) {
                handleNavigation(route);
            }
        });
    });

    // Add hover effects to cards (separate from click handlers)
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px) scale(1.02)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });

    // Add click effects to buttons
    const buttons = document.querySelectorAll('.card-btn, .icon-btn');
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            // Create ripple effect
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';
            ripple.classList.add('ripple');
            
            this.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });

    // Add parallax effect to hero image
    const heroImage = document.querySelector('.hero-image img');
    if (heroImage) {
        window.addEventListener('scroll', function() {
            const scrolled = window.pageYOffset;
            const rate = scrolled * -0.5;
            heroImage.style.transform = `translateY(${rate}px) scale(1.05)`;
        });
    }

    // Add intersection observer for fade-in animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe all cards for fade-in
    const allCards = document.querySelectorAll('.card');
    allCards.forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(card);
    });

    // Add keyboard navigation support
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Tab') {
            document.body.classList.add('keyboard-navigation');
        }
    });

    document.addEventListener('mousedown', function() {
        document.body.classList.remove('keyboard-navigation');
    });

    // Add focus styles for accessibility
    const focusableElements = document.querySelectorAll('a, button, input, textarea, select');
    focusableElements.forEach(element => {
        element.addEventListener('focus', function() {
            this.style.outline = '2px solid #6366f0';
            this.style.outlineOffset = '2px';
        });
        
        element.addEventListener('blur', function() {
            this.style.outline = 'none';
        });
    });

    // Add smooth loading for images
    const images = document.querySelectorAll('img');
    images.forEach(img => {
        img.addEventListener('load', function() {
            this.style.opacity = '1';
        });
        
        img.addEventListener('error', function() {
            this.style.opacity = '0.5';
        });
    });

    // Add mobile menu toggle (for future use)
    const mobileMenuToggle = document.createElement('button');
    mobileMenuToggle.className = 'mobile-menu-toggle';
    mobileMenuToggle.innerHTML = '<i class="fas fa-bars"></i>';
    mobileMenuToggle.style.display = 'none';
    
    const navLeft = document.querySelector('.nav-left');
    if (navLeft && window.innerWidth <= 768) {
        mobileMenuToggle.style.display = 'block';
        navLeft.appendChild(mobileMenuToggle);
    }

    // Add window resize handler
    window.addEventListener('resize', function() {
        if (window.innerWidth <= 768) {
            mobileMenuToggle.style.display = 'block';
        } else {
            mobileMenuToggle.style.display = 'none';
        }
    });

    // Add performance optimization
    let ticking = false;
    function updateParallax() {
        if (heroImage) {
            const scrolled = window.pageYOffset;
            const rate = scrolled * -0.5;
            heroImage.style.transform = `translateY(${rate}px) scale(1.05)`;
        }
        ticking = false;
    }

    window.addEventListener('scroll', function() {
        if (!ticking) {
            requestAnimationFrame(updateParallax);
            ticking = true;
        }
    });

    // Add error handling for failed image loads
    window.addEventListener('error', function(e) {
        if (e.target.tagName === 'IMG') {
            console.warn('Image failed to load:', e.target.src);
        }
    }, true);

    console.log('AI Fashion Platform loaded successfully!');
});

// Add CSS for ripple effect
const style = document.createElement('style');
style.textContent = `
    .ripple {
        position: absolute;
        border-radius: 50%;
        background: rgba(99, 102, 240, 0.3);
        transform: scale(0);
        animation: ripple-animation 0.6s linear;
        pointer-events: none;
    }

    @keyframes ripple-animation {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }

    .keyboard-navigation *:focus {
        outline: 2px solid #6366f0 !important;
        outline-offset: 2px !important;
    }
`;
document.head.appendChild(style); 