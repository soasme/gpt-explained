namespace :book do

  version_string = `git describe --tags --abbrev=0 2>/dev/null`.chomp
  if version_string.empty?
    version_string = '0'
  else
    versions = version_string.split('.')
    version_string = versions[0] + '.' + versions[1] + '.' + versions[2].to_i.next.to_s
  end
  date_string = Time.now.strftime('%Y-%m-%d')
  params = "--attribute revnumber='#{version_string}' --attribute revdate='#{date_string}'"
  header_hash = `git rev-parse --short HEAD`.strip

  pkg_config_path = [
    "/opt/homebrew/lib/pkgconfig",
    "/opt/homebrew/opt/pango/lib/pkgconfig",
    "/opt/homebrew/opt/cairo/lib/pkgconfig",
    "/opt/homebrew/opt/gdk-pixbuf/lib/pkgconfig",
    ENV["PKG_CONFIG_PATH"],
  ].compact.join(":")

  def check_contrib
    if File.exist?('book/contributors.txt')
      current_head_hash = `git rev-parse --short HEAD`.strip
      header = `head -n 1 book/contributors.txt`.strip
      header_hash = header.scan(/[a-f0-9]{7,}/).join

      if header_hash == current_head_hash
        puts "Hash on header of contributors list (#{header_hash}) matches the current HEAD (#{current_head_hash})"
      else
        puts "Hash on header of contributors list (#{header_hash}) does not match the current HEAD (#{current_head_hash}), refreshing"
        sh "rm book/contributors.txt"
        Rake::Task['book/contributors.txt'].reenable
        Rake::Task['book/contributors.txt'].invoke
      end
    end
  end

  def cleanup_stem_images
    FileList['**/stem-*.png'].each do |file|
      rm_f file
    end
  end

  desc 'build basic book formats'
  task :build => [:build_html, :build_epub, :build_fb2, :build_mobi, :build_pdf] do
    begin
      Rake::Task['book:check'].invoke
    ensure
      cleanup_stem_images
    end
  end

  desc 'build basic book formats (for ci)'
  task :ci => [:build_html, :build_epub, :build_fb2, :build_mobi, :build_pdf] do
    begin
      Rake::Task['book:check'].invoke
    ensure
      cleanup_stem_images
    end
  end

  desc 'generate contributors list'
  file 'book/contributors.txt' do
      puts 'Generating contributors list'
      sh "echo 'Contributors as of #{header_hash}:\n' > book/contributors.txt"
      sh "git shortlog -s HEAD | grep -v -E '(dependabot)' | cut -f 2- | sort | column -c 120 >> book/contributors.txt"
  end

  desc 'build HTML format'
  task :build_html => 'book/contributors.txt' do
      check_contrib()

      puts 'Converting to HTML...'
      sh "bundle exec asciidoctor #{params} -a data-uri gpt-explained.asc"
      puts ' -- HTML output at gpt-explained.html'
  end

  desc 'build Epub format'
  task :build_epub => 'book/contributors.txt' do
      check_contrib()

      puts 'Converting to EPub...'
      begin
        sh "PKG_CONFIG_PATH='#{pkg_config_path}' bundle exec asciidoctor-epub3 -r mathematical -r asciidoctor-mathematical #{params} gpt-explained.asc"
        puts ' -- Epub output at gpt-explained.epub'
      ensure
        cleanup_stem_images
      end
  end

  desc 'build FB2 format'
  task :build_fb2 => 'book/contributors.txt' do
      check_contrib()

      puts 'Converting to FB2...'
      sh "bundle exec asciidoctor-fb2 #{params} gpt-explained.asc"
      puts ' -- FB2 output at gpt-explained.fb2.zip'
  end

  desc 'build Mobi format'
  task :build_mobi => 'book/contributors.txt' do
      check_contrib()

      puts "Converting to Mobi (kf8)..."
      sh "bundle exec asciidoctor-epub3 #{params} -a ebook-format=kf8 gpt-explained.asc"
      puts " -- Mobi output at gpt-explained.mobi"
  end

  desc 'build PDF format'
  task :build_pdf => 'book/contributors.txt' do
      check_contrib()

      puts 'Converting to PDF... (this one takes a while)'
      begin
        sh "PKG_CONFIG_PATH='#{pkg_config_path}' bundle exec asciidoctor-pdf -r mathematical -r asciidoctor-mathematical #{params} gpt-explained.asc 2>/dev/null"
        puts ' -- PDF output at gpt-explained.pdf'
      ensure
        cleanup_stem_images
      end
  end

  desc 'Check generated books'
  task :check => [:build_html, :build_epub] do
      puts 'Checking generated books'

      sh "htmlproofer gpt-explained.html" if system('which htmlproofer > /dev/null 2>&1')
      sh "epubcheck gpt-explained.epub" if system('java -version > /dev/null 2>&1')
  end

  desc 'Run every Python code example used by the book'
  task :run_code do
      sh "python3 src/python/run_book_code.py"
  end

  desc 'Clean all generated files'
  task :clean do
    begin
        puts 'Removing generated files'

        FileList['book/contributors.txt', 'gpt-explained.html', 'gpt-explained-kf8.epub',
                 'gpt-explained.epub', 'gpt-explained.fb2.zip', 'gpt-explained.mobi',
                 'gpt-explained.pdf'].each do |file|
            rm file

            rescue Errno::ENOENT => e
              begin
                  puts e.message
                  puts 'Error removing files (ignored)'
              end
        end
        cleanup_stem_images
    end
  end

end

task :default => "book:build"
