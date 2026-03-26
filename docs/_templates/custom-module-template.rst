{% block title %}
{% set prefix_mapping = ({"pp": "Preprocessing", "tl": "Tools", "pl": "Plotting", "utils": "Utilities"}) %}
{% set module_name = fullname | replace("crested.", "") %}
{%- if module_name in prefix_mapping -%} {{ prefix_mapping[module_name] }}: {% endif %}``{{ module_name }}``
{{ "=" * 100 }}
{% endblock %}


.. automodule:: {{ fullname }}

   {% block modules %}
   {% if modules %}
   .. rubric:: Submodules

   .. autosummary::
      :toctree:
      :template: custom-module-template.rst
      :recursive:
   {% for item in modules %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}


   {% block attributes %}
   {% if attributes %}
   .. rubric:: Module attributes

   .. autosummary::
      :toctree:
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree:
      :nosignatures:
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
      :toctree:
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}