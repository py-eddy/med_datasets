CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��-V      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N��   max       P�Y�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��7L   max       >V      �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���R   max       @F�\(�     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���
=p�   max       @vy\(�     �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @L@           t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @��           �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���   max       ><j      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B3�      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B48�      �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�_   max       C��J      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?���   max       C���      �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          u      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          O      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       P���      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���{���   max       ?ݽ���v      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��7L   max       >V      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @F�\(�     �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���
=p�   max       @vy\(�     �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @L@           t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�3        max       @��           �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D   max         D      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�`�d��8   max       ?ݽ���v     �  QX               W                  Z      a   '            ;   #                     J      
   
      1               +                     t            8   	   m   (   
   q            ;   -N3�)O09�N�ÂN{P���N�S�N�Y�N�R<N+�>O�h}P:�*N��zP�Y�O]��N�G3N撚O#C�P�?O���N �fN-��N~O
bwN�bN7M�O�	tO�UN��$NΧtN4�P>�vOG�gN���N=��N��Po��OP.wO5\�O��iOp��N`�O��Pj��N���O��OG�P.��N��)PSg`P�N��O�H�O!ѫNW��N_:�Ox�O4�8��7L���
�t��o��`B�D���D���o��o;�o;�o;�o;ě�<o<t�<T��<T��<u<�o<�o<��
<�1<�j<ě�<���<���<�/<�=+=C�=\)=t�=�P=�w=�w=�w=#�
=,1=49X=49X=8Q�=8Q�=@�=D��=aG�=aG�=e`B=u=y�#=�%=�o=��=�C�=��
=��=�1>V�������������������� #,0<DINQQNI=0# ffgt���������tjgffff�������������������������5W\]IG=9)����!$()45BDNPTNHB5)!!!!TX[gt������ytg[[TTTT��}�����������������mipt����|tmmmmmmmmmm����	"/7;=::5/"	��ZUa�����������zgibaZot�����������vwtoooo<eq���� ���za<!#/<HTVZZURH</#!\aimrz}����|zxmdba\\����������������������������������������������
#(/2.& 
����#%%++/1HU[bffdaUHE8#*/0*%�� 	�������������

�����������SOOQTamnuyzzzsmkaYTS,,0<FE=<;0,,,,,,,,,,��������������������"#&-6BOTSYadca[OB6)"�����
"""
�����()/6BOT[a`[[OB66-)((�����
��������������������������������/<B@/
�������+)+**/<HS`daUTHBNH/+yz|���������}zyyyyyy������������������40+5BBFB<54444444444z������������������z
#02<IUTMIF<0#
WVZacmmz��������zmaW������%+&%������"*+&���������������������������������������������|{������
�������733;GHSTVTNHC;777777rnmpt�����������xtrr�������������������)5BDA;-) �����)BN[m����gN5	������7@BA>5���=69BO[[_[ZOB========��������
#%$
������������������������~������������������*,6BOQOKDB76********�������� 	


���������������������������������������������������������������ػx�����������������������������}�x�q�r�x���ʾ̾Ѿ;ʾþ��������������������������s�}�������z�s�n�f�`�f�h�h�s�s�s�s�s�s�
�#�I�_�nœŒņ�b�U�<��
�������������
�.�;�G�H�T�T�U�T�I�G�;�5�.�*�)�'�.�.�.�.���������������������������������������ſ�������
�������ݿܿܿݿ�����L�Y�e�n�n�e�Y�N�L�J�L�L�L�L�L�L�L�L�L�L�H�T�a�b�i�m�o�p�m�g�a�T�;�/�%� �)�2�;�HāĦĿ������ĿĦā�i�_�[�O�?�>�O�[�h�wā�Z�]�[�Z�U�N�F�A�7�5�1�5�7�A�N�S�Z�Z�Z�Z�N�Z�o����������������s�h�d�[�7�<�;�N���������#�$�'��������������������������������������������������������E�E�E�E�E�E�E�FFFFF F$F)F$FFE�E�E�A�M�Z�[�a�`�]�Z�M�A�4�*�(� �#�(�4�4�A�A��(�4�M�f�������s�Z�A�(����������a�m���������������z�m�a�T�H�:�2�8�;�H�a�`�d�m�n�p�m�`�Y�T�P�T�Y�`�`�`�`�`�`�`�`�"�.�;�>�C�;�.�&�"��"�"�"�"�"�"�"�"�"�"����������������������������������������Ƴ������������������ƸƳƲƧƙƚƞƧƳƳ�����������r�r�r�w����������������������ھ���������������ʼܼ޼ݼּʼ������r�f�X�M�I�S�c������������ ����������������������������#�&�)�&�#���
�������������
����������ÿĿǿοĿ������������������������ûлػڻջлû������ûûûûûûûûûÿy����������y�T�;�.���޾���.�F�T�n�y�4�M�Z�f�s�{��x�s�f�Z�M�U�M�A�4�(�%�)�4���������������s�g�d�f�g�s�}�������������#�#�/�<�=�<�<�/�#���#�#�#�#�#�#�#�#�#������������������������������������������������'�-�*��������������l�\�\�a�s������������������ּ˼ͼռּ��ŔŠŭŴŹ������������ŹŭŠŖŔŐŎŏŔ�	��"�/�;�O�^�S�H�?�/�"�	�������������	���������������������������z�o�b�m�z���������������������������������������������Ŀѿݿ�����-�.�*�!����ٿͿȿĿ��ĺ����������ﺰ���������������������#�0�<�B�<�<�0�*�#������#�#�#�#�#�#�!�-�:�F�K�M�H�F�:�9�-�!����
���!�!�y�������������y�t�l�`�[�X�X�Y�\�`�l�s�y�������������Ƴƚ�u�\�Q�h�sƁƚ�������g�t��~�t�g�[�Y�P�[�_�g�g�g�g���B�[�o�zČďČā�m�[�B�-�'���� �����/�;�5�*�&���
��������¯§²�������ѿݿ�������ݿӿѿȿ˿ѿѿѿѿѿѿѿ�EuE�E�E�E�E�E�E�E�E�E�E�E�E�EuEmEjEkEpEu�U�a�zÇÛàìùàÓÈ�z�n�a�U�H�A�H�O�U�����
�����
�����������������������������������ܻۻܻݻ��������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D~D�D�D����'�$������������������ ]  Y M + : B P . 0 ) ^ f  D K  1 R ^ @ M : ! @ J * 6 9 J B P * : 9 U 4 8 I 8 B 8 5 C * \ V V R ?  ' q q E L F    [  s    �  �  �    �  H      �  	q  �  �    \  �  �  a  H  <  6  0  \    3  '  �  f  k  �  �  h  +  j  �  �  �  �  X  �  �  �  =  �  f  �  �  �  �  t  �  �  �  9  �����t���o�o=���;D��;�`B<49X<t�<���=��<#�
=��=@�<D��<�j=t�=��=P�`<�t�<ě�<�j=\)<��<�/=���=8Q�=#�
=0 �=�w=�1=H�9=#�
=<j=,1=��T=�%=q��=��=�\)=D��=�\)>&�y=aG�=�\)=�+=�S�=�O�>.{=��=��P>5?}=�{=�{=�v�>hs><jB%.B%��B
�Bd�B��BbB	y�B<B_;A��B ��B�	BX�B�pA��B�1B)9B$!B7�B/3B�eB��A���B&B3�B5B��BT�B!�B"�SB�;B�B ��B�B[uB�B%��A�J0B�4B�B:-B��B��A�C.B��B-�B��B�MB�B��B�B��B��BS/B��BzBf@B;�B&*EB	�@B< B�7BR�B	w�B@&BN@A���B ��B�4B��B�A�+lBD�B?�B$JCB@
B/m�B B	uA��AB&?�B48�B>�B�B�eB,,B"��B��B��B b�B�ZBC B'�B%FYA��CB��B��BBMBC�B�1A�t�B�_B,T�B[�B�B�B��B��B��B��B�'B>KBA"B�HA�d�@�A�AO�AC
A� Ac;A�s1A���?�_A��#A�UWA�,�A�=aA�j A��6C��JA;��A9YA���Aib
Aa�A��Bñ@�AV�@@��A���A�Av T@��AjƒA?>�A�X�A�A��3A�S�A�A�V�A��[A���Asm�A���@3&
A�c@sDA��BvgA�_�A�[pA���A}d@C�Aǉ�A�~c@�1C��b@��]A��@���AO%WAB��A�}�Ac��A�#�A�?���A��A�,A�	�A�zA�Z�A��C���A;?A9��A���AheAa"�A���B@�+AV��@�!�A�]�A���Au�@�eAk�A?��A�0�A�v2A���A���A��A�p�A���A��Arw>A�PC@4��A�N�@{��A��B�A��CA�}A�%[A|��C��A�[A�w�@�!�C��6@��v               W                  [      b   (            ;   $                     J               2               +                     u            8   
   n   )      q            ;   .               ;                  +      O               +                        #               3               5         #            1            1      3   '                                    '                        ;                                                                     5                     %            1      !   '                     N3�)N�5�N[�{N7��Pt[N�S�N���N�R<N+�>Od��O��N��zP���O�N�G3N撚N���O���N��ZN �fN-��N~N�oN�bN7M�O>.N�dN��$NΧtN4�N�N��N���N=��N��Pbc2O��O5\�O���Op��N`�O�tRP�6N���O��OG�P.��N��)O���P�N��OF�O�4NW��N_:�NңkO4�8  �  x  �  �  �  w  �  T     �  
&  	  -  �  1  `  �  �  k  m  �  �  �  i  F  	�  �  �  �  �  �       �  {  u  �  '  �  �    �  
�  A  �      �  �  4  w  '  �  �  n    ٽ�7L��C��o��`B<����D����o�o��o;ě�=t�;�o=o<�C�<t�<T��<�1<��<��<�o<��
<�1<ě�<ě�<���=Y�<�`B<�=+=C�=}�='�=�P=�w=�w='�=49X=,1=<j=49X=8Q�=D��=��T=D��=aG�=aG�=e`B=u=��=�%=�o=��=�O�=��
=��=�v�>V��������������������##'0<IJMMIG<30'#niht�������tnnnnnnnn��������������������������);><5)���!$()45BDNPTNHB5)!!!!XZ[gt|���tg][XXXXXX��}�����������������mipt����|tmmmmmmmmmm����	"/1;8773/"	�iitz������������ztoiot�����������vwtooooPUh����� ����zUP"#+/<HKOTTMH</(#""\aimrz}����|zxmdba\\����������������������������������������������
#)+*#
�����;211:<HHQUX]]XUKH<;;*/0*%�� 	�������������

�����������QPRTammtxyrma[VTQQQQ,,0<FE=<;0,,,,,,,,,,��������������������((,16BIOTWXXVROB62+(�����
 !!
�������()/6BOT[a`[[OB66-)((�����
��������������������������������
!#%%$#
���./1///4<HIRMIH=<7/..yz|���������}zyyyyyy������������������40+5BBFB<54444444444|������������������|#0<<IFA<70#WVZacmmz��������zmaW������$*$������"*+&����������������������������������������������������������������733;GHSTVTNHC;777777rnmpt�����������xtrr�������������������)5BDA;-) ���)BN[cjppgNB5)�����7@BA>5���=69BO[[_[ZOB========�������

������������������������~������������������*,6BOQOKDB76********��������

����������������������������������������������������������������ػ���������������������������x�w�x�y���������ʾо˾ʾ����������������������������s�z�}�����u�s�q�f�f�f�i�k�s�s�s�s�s�s�
��0�<�I�U�`�b�^�I�<�#��������������
�.�;�G�H�T�T�U�T�I�G�;�5�.�*�)�'�.�.�.�.���������������������������������������ſ�������
�������ݿܿܿݿ�����L�Y�e�n�n�e�Y�N�L�J�L�L�L�L�L�L�L�L�L�L�H�T�^�a�g�l�m�n�m�a�T�H�;�/�(�#�+�5�;�HĚĦĳķļľ��ļĳĦĚč��t�n�o�zāčĚ�Z�]�[�Z�U�N�F�A�7�5�1�5�7�A�N�S�Z�Z�Z�Z�g������������� �������������q�o�h�R�\�g�������������������������������������������������������������������E�E�E�E�E�E�E�FFFFF F$F)F$FFE�E�E�A�M�Z�\�[�Z�T�M�A�4�4�-�4�5�A�A�A�A�A�A�(�4�A�M�Z�k�s�n�^�M�A�4����������(�T�a�m�z���������z�t�m�a�T�H�E�H�I�S�T�T�`�d�m�n�p�m�`�Y�T�P�T�Y�`�`�`�`�`�`�`�`�"�.�;�>�C�;�.�&�"��"�"�"�"�"�"�"�"�"�"����������������������������������������������������������ƳƧƝơƧƳƸ�������������������r�r�r�w����������������������ھ��������������������������������r�f�b�[�a�f�r�������������������������������������������#�&�)�&�#���
�������������
����������ÿĿǿοĿ������������������������ûлػڻջлû������ûûûûûûûûûÿ`�m�y�����������y�t�m�`�T�P�K�T�U�_�`�`�M�S�Z�c�f�k�s�u�s�s�f�Z�M�L�A�?�A�G�M�M���������������s�g�d�f�g�s�}�������������#�#�/�<�=�<�<�/�#���#�#�#�#�#�#�#�#�#������������������������������������������������$�)�'��������������p�a�_�h�������������������ּܼϼмּڼ��ŔŠŭŴŹ������������ŹŭŠŖŔŐŎŏŔ�	��"�/�;�H�T�N�H�<�/�"�	�������������	���������������������������z�o�b�m�z���������������������������������������������ݿ������(�*�&�������ݿҿѿҿտݺ����ɺֺ����� ��ֺ������������������#�0�<�B�<�<�0�*�#������#�#�#�#�#�#�!�-�:�F�K�M�H�F�:�9�-�!����
���!�!�y�������������y�t�l�`�[�X�X�Y�\�`�l�s�y�������������Ƴƚ�u�\�Q�h�sƁƚ�������g�t��~�t�g�[�Y�P�[�_�g�g�g�g�6�B�O�`�l�|Ā�{�t�h�[�O�B�3�)�"���(�6����/�;�5�*�&���
��������¯§²�������ѿݿ�������ݿӿѿȿ˿ѿѿѿѿѿѿѿ�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�EyEsEvE�E��U�a�n�zÇÖßÓÇÆ�z�n�a�U�J�H�C�H�Q�U�����
�����
�����������������������������������ܻۻܻݻ��������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����'�$������������������ ] * D _  : = P . ,  ^ T  D K " & > ^ @ M ? ! @ 5 + 6 9 J 9 > * : 9 S   8 A 8 B %  C * \ V V D ?  ( _ q E F F    [    t    �  �  �  �  H  �  [  �    J  �    �  �    a  H  <    0  \  �    '  �  f    �  �  h  +  :  5  �  k  �  X  5  m  �  =  �  f  �  �  �  �  �  f  �  �  �  �  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  �  �  �  �  �  ~  {  x  u  r  p  o  m  l  k  i  h  g  e  d  n  q  s  u  w  w  v  s  n  f  \  M  ;  '    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  u  i  [  N  @  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  [  E  .    �  =  o  �  �  �  �  �  �  �  �  �  a  .  �  i  �    �  �  w  n  e  \  T  L  D  8  *      �  �  �  �  w  [  E  0    �  �  �  �  �  �  �  �  }  �  �  �  �  �  �  �  �  �  �  �  T  B  /      �  �  �  �  f  N  <    �  �  L  �  �    �     +  6  7  4  !    �  �  �  �  �  _  <    �  �  >  �  �  �  �  �  �  �  o  W  =  !    �  �  �  �  �  W    �  :  �  �  �  	   	g  	�  	�  
  
&  
  
  	�  	�  	-  �  5  �  )    6  M  	  �  �  �  �  �  �  �  �  �  ~  r  g  T  5    �  �  �  �  �  �      �    "    �  �  s  A  �  �  �  f  �  C  �  �  $  Q  v  �  �  �  �  z  X  +  �  �  X  �  �  "  �  �  1  _  1  '        �  �  �  �  �  �  �  �  �    n  \  J  8  &  `  N  <  0  #      �  �  �  �  �  �  �  \  8    �  �  ^    E  [  r  �  �  �  �  �  �  a  $  �  |    �  v    �  <    d  �  �  �  �  �  �  �  �  _  !  �  �    �  2  �  �  �  �    >  H  K  I  I  [  i  b  O  9    �  �  a  �       �  m  k  h  e  b  `  ]  Z  W  T  Q  M  I  E  A  =  9  5  1  -  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  Z  D  /       �   �  �  �  �  �  �  �  �  �  �  m  O  /    �  �  �  x  =  �  �  i  e  `  \  Z  W  W  Z  ]  d  k  r  y    �  �  �  �  �  �  F  A  =  9  4  0  ,  '  #           �   �   �   �   �   �   �  �  Y  �  	  	F  	�  	�  	�  	�  	�  	n  	2  �  �  >  �  �  O  �  j  �  �  �  �  �  �  �  x  g  Q  6  
  �  z  -  �  �  -  �    �  �  �  �  �  �  �  �  �  �  }  e  J  /    �  �  �  u  +  �  �  �  �  |  r  h  ^  R  B  0       �  �  �  Z  *  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  d  �  �  t  /  �      m  %  u  �  �  s  =  �  �    m  �  J    �  �  �  �  �            �  �  �  �  [  1    �  �       	  �  �  �  �  �  �  �  �  �  �  x  i  Z  K  ;  ,    �  �  �  �  �  �  �  y  j  Z  K  C  h  �  �  �  �  �  v  k  {  v  q  m  h  c  _  Y  S  L  F  ?  9  /         �  �  �  k  u  q  d  N  3    �  �  �  Q    �    �  ~  *  �  �  [  �  �  �  �  �  �  �  �  �  �  �  �  z  ^  @    �  �  <  �  '    
  �  �  �  �  �  y  T  .    �  �  �  V  !  �  �  7  n  �  �    s  c  R  >  &  
  �  �  {  +  �  �  (  $  �  K  �  �  �  |  r  a  L  5    �  �  �  b  (  �  �  �  K  �  s    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  d  P  L  v  �  �  �  w  a  B    �  �  �  R    �  �  �  �  =  �  	J  	�  
  
G  
m  
�  
�  
r  
W  
7  
  	�  	�  	;  �  &  0  �  8  �  A  3  &      �  �  �  �  �  �  �  �  �  �  �  y  w  x  y  �  �  {  [  :    �  �  �  �  �  �  \  ,  �  �  r    �  i    �  �  �  �  �  �  v  g  Y  K  :  '      �  �  �  �  �    �  �  �  �  }  b  �  �  �  �  �  c  *  �  k  �  I  ~  �  �  �  �  �  z  ^  A  &    �  �  �  �  w  P  %  �  �  z  =  Q  �  �  #  W  �  �  �  �  �  o    �    
l  	�  r  �  \  p  4  #  	  �  �  |  >    �  �  U  ,  �  �  7  �  �  *  �  H  w  n  e  Y  M  ?  1  !    �  �  �  �  �  �  y  j  [  L  >  �  �  n  �       '    �  �  L  �    2  �  �  
�  	>  W  B     �  �  x  _  D  (    �  �  �  o  5  �  �  7  �  _  �  N  �  �  �  �  �  �  �  �  �  |  r  h  `  Y  R  J  E  ?  :  5  n  [  G  1       �  �  �  �  h  F  #    �  �  �  �  �  z  �  �        �  �  y  5  �  �  9  �    P  
m  	}  z  f  '  �  �  x  ?    
�  
�  
M  	�  	�  	.  �  J  �  S  �  �  �  h  L