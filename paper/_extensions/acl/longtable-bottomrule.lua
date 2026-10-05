-- Move \\bottomrule out of \\endlastfoot so it is typeset before Quarto's
-- trailing \\caption row rather than after it. Without this, a longtable with
-- tbl-cap-location: bottom draws its closing rule underneath the caption text.
local function fix(s)
  if not s:find("\\begin{longtable}", 1, true) then return nil end
  local new, n = s:gsub("\\bottomrule%s*\n(\\endlastfoot)", "%1")
  if n == 0 then return nil end
  local out, m
  if new:find("\\caption", 1, true) then
    out, m = new:gsub("(\\caption)", "\\bottomrule\n%1", 1)
  else
    out, m = new:gsub("(\\end{longtable})", "\\bottomrule\n%1", 1)
  end
  if m == 0 then return nil end
  return out
end

function RawBlock(el)
  if el.format ~= "tex" and el.format ~= "latex" then return nil end
  local out = fix(el.text)
  if out then return pandoc.RawBlock(el.format, out) end
end
