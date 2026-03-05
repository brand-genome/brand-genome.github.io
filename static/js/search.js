/**
 * Brand Identity Knowledge Graph - Search Functionality
 */

(function() {
    'use strict';

    // Search data (will be populated by Hugo)
    let searchData = null;
    let searchIndex = [];

    // DOM elements
    let searchInput = null;
    let searchResults = null;

    // Initialize search
    function init() {
        searchInput = document.getElementById('brand-search');
        searchResults = document.getElementById('search-results');

        if (!searchInput || !searchResults) return;

        // Load search index
        loadSearchIndex();

        // Event listeners
        searchInput.addEventListener('input', handleSearch);
        searchInput.addEventListener('focus', () => {
            if (searchInput.value.length >= 2) {
                showResults();
            }
        });

        // Close results on outside click
        document.addEventListener('click', (e) => {
            if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
                hideResults();
            }
        });

        // Keyboard navigation
        searchInput.addEventListener('keydown', handleKeyboard);
    }

    // Load search index from JSON
    async function loadSearchIndex() {
        try {
            const response = await fetch('/index.json');
            if (response.ok) {
                searchData = await response.json();
                searchIndex = searchData.brands || searchData || [];
            }
        } catch (error) {
            console.log('Search index not available:', error);
        }
    }

    // Handle search input
    function handleSearch() {
        const query = searchInput.value.toLowerCase().trim();
        
        if (query.length < 2) {
            hideResults();
            return;
        }

        const results = searchIndex.filter(item => {
            const title = (item.title || '').toLowerCase();
            const sectors = (item.sectors || []).join(' ').toLowerCase();
            const regions = (item.regions || []).join(' ').toLowerCase();
            const description = (item.description || '').toLowerCase();
            
            return title.includes(query) || 
                   sectors.includes(query) || 
                   regions.includes(query) ||
                   description.includes(query);
        }).slice(0, 10); // Limit to 10 results

        displayResults(results, query);
    }

    // Display search results
    function displayResults(results, query) {
        if (results.length === 0) {
            searchResults.innerHTML = '<div class="no-results">No brands found</div>';
            showResults();
            return;
        }

        const html = results.map((item, index) => {
            const highlightedTitle = highlightMatch(item.title || 'Unknown', query);
            const sector = item.sectors && item.sectors[0] ? item.sectors[0] : '';
            const region = item.regions && item.regions[0] ? item.regions[0] : '';
            
            let meta = [];
            if (sector) meta.push(sector);
            if (region) meta.push(region);
            
            return `
                <a href="${item.url || '#'}" class="search-result-item" data-index="${index}">
                    <span class="result-title">${highlightedTitle}</span>
                    ${meta.length > 0 ? `<span class="result-meta">${meta.join(' · ')}</span>` : ''}
                </a>
            `;
        }).join('');

        searchResults.innerHTML = html;
        showResults();
    }

    // Highlight matching text
    function highlightMatch(text, query) {
        if (!query) return text;
        const regex = new RegExp(`(${escapeRegex(query)})`, 'gi');
        return text.replace(regex, '<mark>$1</mark>');
    }

    // Escape special regex characters
    function escapeRegex(str) {
        return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    // Show results dropdown
    function showResults() {
        searchResults.classList.add('active');
    }

    // Hide results dropdown
    function hideResults() {
        searchResults.classList.remove('active');
    }

    // Keyboard navigation
    function handleKeyboard(e) {
        const items = searchResults.querySelectorAll('.search-result-item');
        const activeItem = searchResults.querySelector('.search-result-item.active');
        let activeIndex = -1;

        items.forEach((item, index) => {
            if (item.classList.contains('active')) {
                activeIndex = index;
            }
        });

        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                if (activeIndex < items.length - 1) {
                    if (activeItem) activeItem.classList.remove('active');
                    items[activeIndex + 1].classList.add('active');
                    items[activeIndex + 1].scrollIntoView({ block: 'nearest' });
                }
                break;

            case 'ArrowUp':
                e.preventDefault();
                if (activeIndex > 0) {
                    if (activeItem) activeItem.classList.remove('active');
                    items[activeIndex - 1].classList.add('active');
                    items[activeIndex - 1].scrollIntoView({ block: 'nearest' });
                }
                break;

            case 'Enter':
                e.preventDefault();
                if (activeItem) {
                    window.location.href = activeItem.href;
                } else if (items.length > 0) {
                    window.location.href = items[0].href;
                }
                break;

            case 'Escape':
                hideResults();
                searchInput.blur();
                break;
        }
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();

/* Additional styles for search results */
(function() {
    const style = document.createElement('style');
    style.textContent = `
        .search-result-item {
            display: flex;
            flex-direction: column;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid #f0f0f0;
            transition: background-color 0.1s;
        }
        .search-result-item:last-child {
            border-bottom: none;
        }
        .search-result-item:hover,
        .search-result-item.active {
            background-color: #f5f5f5;
            text-decoration: none;
        }
        .result-title {
            font-weight: 500;
            color: #333;
        }
        .result-meta {
            font-size: 0.8125rem;
            color: #666;
            margin-top: 0.25rem;
        }
        .search-result-item mark {
            background: #fff3cd;
            padding: 0 2px;
            border-radius: 2px;
        }
        .no-results {
            padding: 1rem;
            text-align: center;
            color: #666;
        }
    `;
    document.head.appendChild(style);
})();
