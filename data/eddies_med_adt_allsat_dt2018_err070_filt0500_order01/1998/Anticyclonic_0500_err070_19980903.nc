CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�ě��S�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N Y�   max       P��,      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �49X   max       >I�      �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @F�����     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���R    max       @v�z�G�     �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P            t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��@          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       >l�D      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B/&�      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��}   max       B/�      �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ? �/   max       C��      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�n   max       C��      �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N Y�   max       P``T      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�MjOw   max       ?�n.��2�      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��w   max       >I�      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @F�\)     �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�
=p��    max       @v�fffff     �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @P            t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��          �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Eq   max         Eq      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��PH�   max       ?�m\����     �  QX         
   7         Q         )               	         X                  N         �   Q   P   U         ,   	   4   >      ]      .   &   �   X                  7   *            �   2      )O.��N Y�N�;�PbfWNk�NV\zP��,O�9�N.x�Ol�NJo�Oo�O�P.&N�S�NM�CNa$�Pl�O�OZU�N�LMN��N�}APK=�N�ǯN��P!�UP�B�Pp��P���OO�ON�SOq��N�GTOaNwO��/O�[P~mN��P^�(O�r�P(H;P1��N��N�Q@N�)�O�9�N�ԑO��tOJ=ZO(/.N��8N�:�O�P�OX�NR�OMk�49X��t��T���49X�#�
�t���o%   %   <#�
<49X<T��<u<�o<�o<���<��
<�1<�j<ě�<���<���<���<���<�h<��=o=o=o=\)=\)=\)=�P=�w=#�
='�=49X=8Q�=D��=H�9=H�9=L��=]/=aG�=aG�=ix�=q��=}�=�%=��=��=\=���=���=��=��>I���������������������eegtvywtogeeeeeeeeee������������������������
/BGJU`e</��� #%,/<@C></#        �����������������������	*<^jb]axwaH<������������������������������������������#<HRY]\UOKH<9/'%#^\Zanzzwnja^^^^^^^^�����������������qs���������������uqKJMTaz��������zma\TK.+/<AIUYbkebXUI@<30.?ABGOUW[\\[[VQOHFB??"+/087/"������)6<@?5-"�����
*/6750*%��������������������������������������������������������������)/1)
�������� �O[jmmh[O6	��fanrz���������znffff������
��������kow��������������zpk����������������������������������������)BPSTO<*������OX[t����������|tnd[O|�����������������|������������������������������������������������������������������������������������������bbdbz������������mab8544<BHOUUWVUH?<8888������)5@@9)����������
#<IPPI=0�����	9B[hiellppdO)	���)[hwthUPMB)�
#%17<<=<9/#!
./19<HJUW[]aUH@<1/..����������������� !/<?></#
������������������������������(9:6*�������NNV[gt��������tge[UN��������������������BKN[gtu}~tg[QNBBBBBB��

�����������������������

����������

����������;<CHKOOKHE?<;;;;;;;;#,/12/.#

###���
����
����������ĿĹ����������������������������������������������������� �������������������5�N�d�v�}�|�m�Z�N�(�����޿пϿ����5D�D�D�EE
EED�D�D�D�D�D�D�D�D�D�D�D�D��)�6�B�L�B�B�6�)�'��)�)�)�)�)�)�)�)�)�)�"�T�������������z�T�F�"�������������	�"���(�<�X�]�]�U�N�A�5�+������������(�5�A�E�D�A�5�(�&�#�(�(�(�(�(�(�(�(�(�(���������������������������������a�n�z�}Ä��z�r�n�b�a�`�a�a�a�a�a�a�a�a�������ûлػܻ�ܻϻû������������������M�Z�f�s����x�f�Z�M�4������(�6�A�M�����	����	�����������g�a�i�z��������r�����������������������|�o�f�d�f�i�r�z�������������������}�z�m�a�W�a�m�q�z�z�a�e�m�u�z���z�m�a�[�V�[�a�a�a�a�a�a�a�a�����0�I�{őřŖŎ�{�n�U�0�������������m�y������������������y�m�d�`�[�\�`�a�m�/�;�H�O�T�]�a�f�a�a�T�H�;�/�#��� �#�/��'�.�4�@�M�P�M�@�;�4�'���������f�r�������r�f�`�b�f�f�f�f�f�f�f�f�f�f�f�s�s�s�r�o�l�f�Z�T�R�W�Z�a�f�f�f�f�f�f�׾��"�;�7�(�� ��ʾ��������������������
��������
�������������������������
������
�����������������������tčĦĹ��������Ŀĳč�t�i�O�B�:�7�E�[�t������-�)�-�>�6�����Ó�z�l�g�o�zàì�޽н���A�Z�d�`�M�?�=�4�(�����|���������������������������ƳƁ�h�V�hƎƧ�������H�U�W�V�[�a�U�T�<�/�'����#�+�/�<�E�H�s�����������������������������x�s�g�g�sFFF1F=FJFXFgFkFlFcFVFJF=F1F$FFE�FF�������������ݿѿȿĿ��Ŀѿݿ���Çàïùû��ûùìàÓÇÃ�z�w�t�v�z�Ç�ܻ�������������ܻлû������ɻܹϹܹ���������#�������ٹйϹ̹�ŭ������������� ����������ŹŭŤŘōœŭ�.�;�G�T�[�T�P�G�;�;�.�"�!��"�&�.�.�.�.�(�5�A�F�I�I�Q�M�C�(����ǿɿؿ����(��������������׺ɺʺѺպغ����M�f�r���������������r�Y�@�4�'� ��&�4�M��-�H�b�j�k�q�h�n�_�S�M�-�!��ٺ�����f�s���������s�f�\�Z�M�C�M�T�Y�Z�]�c�f��!�(�4�?�6�4�,�(������������������������������������������������������u�r�g�N�J�B�B�H�Q�[�g�t�ʼּؼ߼�ּܼʼɼ��������¼ʼʼʼʼʼʽ������������|�y�l�`�S�I�E�B�C�G�T�f�l������ �#�!���������������������������'�1�6�G�O�S�O�B�6�6�)�"�������"�#�'�(�$�"��	������	�������ǮǭǧǡǔǈǆǀǅǈǔǡǢǮǮǮǮǮǮǮD{D�D�D�D�D�D�D�D�D�D�D�D�D�D�DrDhDgDpD{E�E�E�E�E�E�E�E�E�E}EyEwEvE{E�E�E�E�E�E���������������ŹŭŹ���������������������O�C�;�6�*� ������*�6�C�I�O�U�O�O�O R \ U 4 > C 5 8 - = G 1 G g a b H M < , b ; u > d & L F D H ` I V Z - D [ ? O 6 q 6 @ h 6 W ] ] } : � X # & F h     �    +  �  �  ]  �  V  =  �  v  G  �  [  V  �  �     0  �  �  5  �  �  O      �  E  �  �      �  �  �  o  �  �  �  d  �  #  @      L  �  (  �    (  �  9  �  U  -�����u�D��=49X%   <�o=���<�9X;o=L��<u=��=\)=,1<ě�<ě�<���=�;d<�='�=\)<�h<��=���=�P=8Q�>"��=�l�=�`B=�F=��='�=��
=@�=�^5=��=ix�>I�=q��=\=� �>6E�>hs=��=�O�=�O�=�E�=��==��=��=�`B=>l�D>��=�l�>6E�B��B	�`B��BM�BO�B`�B��B��B��B�7B��B#+*B|�A���B&�dB�UA���ByB/&�B�=B!�B"
�BچB�;B�'B��B8�B-B!�KBC�BN�BV�B��B
�B!�B��B�NB}B&�B§B$k�B�?B�B��B2�B�bB.5BΔB-�$B	��B4�B�CBItB��Bm]BD�BgB�B	��B��B��BV�BGB�aB�~B��B�B�`B#@}B�(A� B&Y�B��A��}B@B/�B��B!��B":BGDB��B^B��B �B��B!� B��B}+B� B��B
ͰB":lB�8B�#B�`B?�B�B$�B��B��Bn�B=B�SB�]B�3B-�iB	\
B�<B	BeB@B��B��B@bB?�A���A��%AЬ�A�7�C�DnA�u:A�Z&A���A��AAҮA��&@�s�A;dXA�-e@�`�A�dCA�%�A릥Am�A��2@ʎ\@��A@L�AR�A�߭A���AްuAϟ�A,��B��A�X#A���C��A}ܖA��\@�y? �/A���AcfA�>1@FZm@ڱ,@s��AAʆA4� A���A�(�A M�A��AӿA�^�A�OMBX�C���C��A���B OA��A��Aв�A�x�C�E�A�kAA��iA��YA��'A�x(A�S�@�A�A9 �A�(y@��CA�|&A��A��AlM�A�yb@�IV@�f�A?'/AR۱A���A���Aޅ�AЂ�A,�}B�A�t+A�&�C��A~�\Aʦw@���?�nA��$AaQ�A�q�@L��@�n@s��AB�,A5�A�|^A���@�)VA�>A�}�A�!�A�~�B�,C�׵C��A�\5B 9�            8         Q         )               	         Y                  O   	      �   R   Q   U         ,   	   4   ?      ]      /   &   �   Y                  7   *            �   3      *            5         E                  #   +            1                  3         +   ;   7   ;                  !      %      /   %   +   1                  '                                    -                              )            )                           !   3   !   #                              /   %      #                  !                        NԎ�N Y�N�D�P4�ZNk�NV\zO�Q�OR�`N.x�N�ͮNJo�N���O��=PsdN�S�NM�CNa$�P)�cO�OG[WN�LMN��N�}AO���N�ǯN��O�=P``TO��JO��N�ʹN�SO�}N�GTO8��O���N�w�O���N��PQw�O�r�Oq'�O�7vN�<N��MN�)�O<�N�ԑO��oOJ=ZO(/.N��8N�:�O]R�OX�NR�OMk    ;       �      �  y  �  �  �  -  �  H  $  f  �  b  (    Q  �  Y    \    	~  2  v  �  �  �  �  	�  	�  �  �  1  3  a    
W  �  ]    �  �  �  	Y  �    �  G  V  �  _��w��t��49X�o�#�
�t�=49X;��
%   <���<49X<���<���<�C�<�o<���<��
=��<�j<���<���<���<���=ix�<�h<��=�t�=P�`=�%=�+=8Q�=\)=P�`=�w=@�=L��=<j=��=D��=P�`=H�9=���=��P=e`B=e`B=ix�=�+=}�=��=��=��=\=���>J=��=��>I���������������������eegtvywtogeeeeeeeeee���������������������������
/BDPNG</� #%,/<@C></#        ������������������������
#/47>A@</#������������������������������������������.+)+/<FHOTRHG<7/....^\Zanzzwnja^^^^^^^^�����������������}}����������������LKLOTaz��������zmaVL.+/<AIUYbkebXUI@<30.?ABGOUW[\\[[VQOHFB??"+/087/"������)0101"����
*/6750*%��������������������������������������������������������������)/1)
������)6BOX^][TOB6 fanrz���������znffff������
��������wvw|��������������{w����������������������������������������)5<EIJD5)���{ux�������������{{{{|�����������������|������������������������������������������������������������������������������������ ��������ytsw��������������zy8544<BHOUUWVUH?<8888�����)5>>7)����������
#<IPPI=0�����)6BOTXWSOMB6)���)>MPPGGB6)��
#$/16:/&#
//2<<GHUV[\^UHC<4///�����������������
$/996/#
�������������������������������.2-�����NNV[gt��������tge[UN��������������������BKN[gtu}~tg[QNBBBBBB��

�����������������������
����������

����������;<CHKOOKHE?<;;;;;;;;#,/12/.#

###�������������������������������������������������������������������������������������������������������(�5�A�Q�[�k�v�r�d�N�(������ٿڿ��D�D�D�EE
EED�D�D�D�D�D�D�D�D�D�D�D�D��)�6�B�L�B�B�6�)�'��)�)�)�)�)�)�)�)�)�)�;�H�T�a�i�o�p�j�a�H�/�"�����	��"�;���(�3�A�N�W�W�N�A�5�(����� ����(�5�A�E�D�A�5�(�&�#�(�(�(�(�(�(�(�(�(�(����������������������������������a�n�z�}Ä��z�r�n�b�a�`�a�a�a�a�a�a�a�a�������ûллԻлʻû��������������������A�M�Z�]�f�u�{�n�f�Z�M�A�4�(�����(�A�������	����	�������������s�l�|������r�����������������������|�o�f�d�f�i�r�z�������������������}�z�m�a�W�a�m�q�z�z�a�e�m�u�z���z�m�a�[�V�[�a�a�a�a�a�a�a�a�����
�0�U�{ŊōŇ�}�n�b�I�0������������m�y������������������y�m�d�`�[�\�`�a�m�/�;�H�M�T�[�`�_�T�P�H�;�/�%���!�"�$�/��'�.�4�@�M�P�M�@�;�4�'���������f�r�������r�f�`�b�f�f�f�f�f�f�f�f�f�f�f�s�s�s�r�o�l�f�Z�T�R�W�Z�a�f�f�f�f�f�f�ʾ׾����������ʾ��������������������
��������
�������������������������
������
�����������������������tāčĚīĵĻ����ĿĵĦā�t�l�`�Z�[�h�t�������-�.������ì�y�q�q�zÇì����Ľнݽ���� �'�$������ݽ���������������������������������ƳƗƈƅƎƚƧ�����/�<�H�O�R�L�H�<�5�/�,�$�#�!�#�)�/�/�/�/�s�����������������������������x�s�g�g�sF$F1F=FGFJFVF\F_FZFVFJF=F1F$F#FFFFF$�������������ݿѿȿĿ��Ŀѿݿ���ÇÓàèì÷ùýùìàÓÇ�{�z�w�y�~ÄÇ���������	������ܻлǻ»��û̻ܻ�ܹ��������
������ܹܹԹԹܹܹܹ�ŭŹ����������������������ŹŴŭŪŦūŭ�.�;�G�T�[�T�P�G�;�;�.�"�!��"�&�.�.�.�.�(�5�A�G�G�N�J�?�(����Ϳɿ˿ٿ����(��������������׺ɺʺѺպغ����@�M�Y�f�r�~�����z�r�f�Y�M�B�@�6�3�3�=�@�!�-�:�U�_�f�_�S�F�:�!������������!�f�s���������t�s�f�Z�V�Z�[�^�d�f�f�f�f���(�4�5�4�4�*�(������ �����������������������������������������������t�n�g�[�N�I�G�L�U�[�g�t�ʼּؼ߼�ּܼʼɼ��������¼ʼʼʼʼʼʽl�y�������������{�w�l�`�S�J�G�E�F�L�Y�l���� �#�!���������������������������'�1�6�G�O�S�O�B�6�6�)�"�������"�#�'�(�$�"��	������	�������ǮǭǧǡǔǈǆǀǅǈǔǡǢǮǮǮǮǮǮǮD{D�D�D�D�D�D�D�D�D�D�D�D�D�D�DzDoDnDzD{E�E�E�E�E�E�E�E�E�E}EyEwEvE{E�E�E�E�E�E���������������ŹŭŹ���������������������O�C�;�6�*� ������*�6�C�I�O�U�O�O�O F \ 6 2 > C , F - + G - C a a b H V < , b ; u . d & @ G +  5 I I Z & B D 5 O 7 q   U 4 W j ] s : � X # ' F h     �    �  <  �  ]  �  �  =  �  v  �  4     V  �  �  x  0  �  �  5  �  �  O    �  �  �    �    5  �  �  e  �  V  �  �  d  �  �  �  �    �  �    �    (  �  �  �  U  -  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  Eq  �  �              �  �  �  �  ~  M    �  v    �  �  ;  B  H  O  U  \  b  i  p  w  ~  �  �  �  �  �  �  �  �  �  �  �            �  �  �  �  �  �  v  v  }  �  �  �  g  �  �  �    �  �  �  �  �  �  o  H    �  }        �  �  �  �  �  �  �  ~  g  O  5    �  �  �  �  g  A    �  �  q    �  +  >  Q  e  y  �  �  �  �  �  �  �      a  �    h  �    S  �  �  �  �  �  �          �  �  k  �  &  �  Q  �  �  �  �  �  �  �  �  �  �  q  Q  !  �  �  �  g  P  C  '  y  w  u  s  r  p  n  l  j  h  g  f  e  c  b  a  `  _  ^  ]  P  �  �    .  U  t  �  ~  q  V  &  �  �  6  �  i     �  9  �  }  r  g  ]  Q  A  1  "      �  �  �  �  �  �  �  �  �  o  o  v  |  �  �  u  _  S  I  7  %    �  �  I  �  h  �  B      "  )  -  *          �  �  �  �  �  �  P    �  :   z  �  �  �  �  �  �  m  X  C  +    �  �  �  X    �  1  �  A  H  @  8  2  ,  %        ,  3  2  0  *  #      �  �  [  $  %  %  &  $    	  �  �  �  �  �  �  �  o  [  F  1      f  ^  V  N  <  (    �  �  �  �  �  �  c  @    �  �  �  �  �  �  �  �  �  �  �  p  L  a  �  �  L    �  (  {  �  }   �  b  T  E  7  (    
  �  �  �  �  �  �  �  j  N  2     �   �    '  &  #        
  �  �  �  �  �  �  f  6  �  �  s          �  �  �  �  �  �  z  f  Q  =  (    �  �  �  �  �  Q  N  L  I  F  C  >  :  5  1  ,  '  #                �  �  �  �  �  v  j  ^  Q  E  9  -      �  �  �  �    a  -  �  �  �    6  K  V  X  K  /    �  �  >  �    I  :  �          	    �  �  �  �  �  �  �  t  X  2    �  �  �  \  \  Y  U  R  I  =  -    �  �  �  �  N  �  �  -  �    X  k  :  �  �  �        �  �  P    �  6  �  �  �  
    �  �  	  	=  	e  	{  	x  	Y  	0  �  �  D  �  l  �  ;  |  �  :  �  �  �  �  �  �  �    )  1  0  (      �  �  �  '  �  �  �  �  �      ;  Y  i  p  v  r  `  D    �  e  �  c  �  �  �  �  z  �    O  �  �  �  �  �  �  �  Z  *  �  �  P  �  �  S    �  z  s  l  c  Y  O  @  .      �  �  �  �  �  x  c  O  :  m  �  �  �  �  �  �  �  �  �  L    �  p    �  ^  �  S  �  �  �  �  �  �  �  �  �  ~  d  G  )    �  �  �  �  f  -  �  	�  	�  	�  	�  	�  	�  	�  	J  	  �  e  �  �    �    ~  �  �  �  	�  	�  	�  	�  	�  	�  	�  	`  	  �  k  �  �  �  `  �  �  9  �   �  f  �  �  �  �  �  �  �  �  r  U  4    �  �  �  a  2  
  �  
w  9  �  )  c  |  �  w  Y  +  �  �  _  
�  
m  	�  �  �  ,  �  1      �  �  �  �  �  {  a  G  5  &    �  �  �  x  +  �  (  3  *    �  �  �  �  c  <    �  �  g    �  F  �  )   �  a  N  C  .      �  �  �  �  o  A    �  �  '  �  W    �  
:    �    f  �  �  �       �  �  a  �  B  
d  	]  �    �  	O  	v  	�  
A  
V  
T  
C  
   	�  	~  		  �     o  �    [  �  |  �  q  �  �  �  x  a  G  *  	  �  �  �  i  ?    �  �  �  q  @  K  X  X  P  L  <  )        �  �  �  �  �  Z    �  �  L    
  �  �  �  �  �  �  x  c  N  ;      �  �  �  5  �  j  �  �  �  �  �  �  �  �  �  �    P    �  �  W  �  |  �  C  �  �  �  �  �  w  a  I  ,    �  �  �  �  j  F  �  �  �    �  �  �  �  �  �  �  E  �  �    �  <  �  �  X  �    B  u  	Y  	P  	A  	-  	  �  �  �  T    �  �  B  �  A  �  �  /  ]    �  �  �  �  W    �  �  �  �      �  �  �  �  �  �  l  W    �  �  �  �  �  p  [  F  .    �  �  �  ^    �  )  �    �  �  x  O  #  �  �  �  U     �  �  �  X  #  �  �  p  '  �  p  �    <  F  1    �  �  *  �  �      �  >  �  	  �    V  J    
�  
�  
�  
�  
�  
U  
  	�  	P  �  S  �    \  �  �  #  �  �  �  �  �  �  v  S  0    �  �  �  o  A    �  �  @   �  _    
�  
�  
�  
c  
&  	�  	�  	P  �  �     n  �  ,  t  �  �  	