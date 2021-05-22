CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�$�/��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Na�   max       Q �      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =�
=      �  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�Q��   max       @E�ffffg     �   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�G�z�    max       @vp(�\     �  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @P�           p  1�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��          �  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       >W
=      �  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��;   max       B,b�      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�V   max       B,Wc      �  4�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�C�   max       C�j      �  5�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?ʨ�   max       C�g�      �  6�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          O      �  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          K      �  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       P�M      �  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���'RTa   max       ?�C,�zy      �  :�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       >o      �  ;�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>޸Q�   max       @E�ffffg     �  <�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�G�z�    max       @vp(�\     �  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @P�           p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���          �  N�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�      �  O�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��$�/   max       ?�C,�zy     �  Pl               z   	      1   	                  7            �            "         
            L      <   "         %   	   !                     
   '   L      9   !      �      I      N��Ne|nO�X�On��Q �N7p�N��O��NL�"N(�N�P�19O	FPNS�PJ�
O��tN�ɳN���P=��N,B�OS�=N8��P"KOJ��NR��OM��O1��NJ�ObXP��O�NhPغO���N-�%O.�P��N͂
O��,N2n�N��GN�O=<�O(O#�N;~EO�?P�fNa�O�wO�x*N�A�O�N+-OX��N���N��g���u�T�����
��o:�o;�o;�o;�`B<t�<49X<D��<D��<D��<T��<u<�o<�C�<�t�<�t�<���<��
<�j<ě�<ě�<���<���<�/<�/<�h<�h<�<�<��<��=o=o=\)=�P=�P=��='�='�='�='�=49X=49X=q��=�7L=�O�=�\)=�hs=���=��w=��=�
={z������������{{{{{{#*0220#/6;@DHKIH/" 	""/#05=GUbigaPI<0,%#�����CWjomrg[5���))26<BOQOBB6.)))))))��������������������`\cm��������������g`TUZanstnmaYUTTTTTTTTghknuz��zngggggggggg���������������������������-;?@C?5)��������
#%#!
�����)'-/<?HJHC</)))))))) !)5BNQSNQOB0)"%6B[glpqog[I5)��������������������KJUXannsxnaUKKKKKKKK;8A[gt����������g[I;��������������������jpx�������������znj��������������������CA<<Tmz��������zaTHC��������������������^anz����znja^^^^^^^^��������������������VSQUXV[hntw{|ytph[V����

�����������  #/<ACB9/#
�~~����������������������

������jgn���������������xj�����(--+)�����������������������������
"$$#
�����������
#('
�������������������������}������������������������������������������������Q[`hmt����{th[QQQQQQ#/3@IQRQHD</-'#����������!).6BO[ahlmh^O@:6-)!�����������������������
#/6;<:1.# �>;?O[jz������|thOFC>"#/<B<8/#����������������� ��)5BOZjqgN5 �����������������������������
 " 
�������������������������������

��������������������������������������������������������������������������������Ľнݽ��ݽٽнĽ������ĽĽĽĽĽĽĽ����z�w�m�T�H�;�/�-�"�(�;�H�a�i�m�t�z�|���r����������������������f�`�^�W�Y�f�k�rƚ��������I�G�$�����ƳƎ�%���L�k�uƚÇÇÓÕÓÏÇÇÆ�z�z�o�zÆÇÇÇÇÇÇ�U�a�j�c�a�U�H�B�H�P�U�U�U�U�U�U�U�U�U�U����������	�������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E����������������������������������������Ѻ������Ⱥĺ����������������������������������I�bŇ����Š�{�b�I�������ĺľ�������a�m�z�}��|�z�y�y�m�f�a�V�T�R�N�T�U�a�a�������������������������������������������
�/�H�n�~ÏÖÍÇ�n�H�����²¿�������������������������������������������ѻ��!�-�7�8�:�<�:�-�!������������a�n�w�z�}�{�z�n�a�^�X�Y�a�a�a�a�a�a�a�a��)�B�]�d�g�b�[�O�?�6�'����������������ʼּ������ּʼƼ�������������������������������� �������������������������#�/�<�=�<�9�/�#���#�#�#�#�#�#�#�#�#�#���
�#�0�H�E�?�6�#�
������������������������(�3�3�,�(�"���������ڿݿ���<�B�B�?�=�<�/�)�'�-�/�7�<�<�<�<�<�<�<�<�A�L�a�f�Z�U�M�A�6�(����
���%�(�3�A�"�/�;�T�a�m�m�p�m�`�T�H�;�/�"�����"�5�A�N�Y�X�N�A�5�1�2�5�5�5�5�5�5�5�5�5�5�g�s�����������������������������y�s�]�g�������� �)�)�%�����ùåÛÜÓÖàì�Ҿʾ׾��������׾ʾ������������������ʻ������л����лû����a�W�S�_�m�x�����f�t���������������s�f�d�Y�V�X�^�_�`�f�B�O�W�[�b�[�O�B�7�<�B�B�B�B�B�B�B�B�B�B�M�Z�f�s�v����s�f�Z�M�K�A�8�4�2�4�;�A�M�����)�B�[�r�o�h�]�W�N�B������������޾(�4�A�M�T�X�X�M�A�4�3�(�&�$�(�(�(�(�(�(�M�Z�f�s�x����{�s�`�Z�M�6�(� ��(�4�>�M����������������������������������������ù����������þùìàÓÒÓÖÝàìòùù�@�E�L�Q�Y�_�`�Y�L�E�@�;�5�<�@�@�@�@�@�@�f�s�����������s�f�Z�M�C�A�=�?�A�M�X�f��!�-�:�;�A�B�:�3�-�!�����������	��-�:�F�K�O�S�Z�_�c�_�S�F�:�-�!����!�-���ûлܻԻлû��������������������������`�m�y���������y�m�`�T�G�;�/�.�$�.�;�G�`�'�M�f�~����r�Y�M�@�4����޻ٻ����'��������� �������������������������������	��/�;�T�a�m�����������z�a�T�H�/�"��	���������Ͽѿ�׿Կ��������y�n�h�m�z��������������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DwDyD����������������y�t�s�y������������������EuE�E�E�E�E�E�E�E�E�E�E�E�E�E{EuEpElEiEu�<�H�U�U�\�a�b�a�U�H�<�4�/�+�/�0�<�<�<�<�ּ������������ּӼҼּּּּּ� U 7 7 R ? O ^ E # 1 D H J ( i [ B D  � @ 8 . ; y R L 3 Y C % > 0 B ' \ 4 6 O G v F ( w ` " C � d T >  V $ 1   �  u  9    	N  w    �  ^  D    �  H  h  �  y    �    �  �  D  �  �  �  �  �  Z  (  �  J  �  ^  W  O  	  �  I  U    �  �  e  �  L    �  y  �  i  �  �  Z  �  �  ������D��<o<D��=���<t�;�`B=T��<u<49X=o>O�<�j<u=�C�=0 �<�<�1>C��<�j=C�<�/=ix�=C�=o=t�=�P=C�=8Q�=���=<j=�Q�=�%=,1=8Q�=�C�='�=�7L=�w=q��=T��=m�h=y�#=q��=P�`=��=�=�o=��m=��=�t�>W
==��T>��=��>$�B
��B%/.A��;B&�uBw�B�SB�B
�BBZ
B�3B!��BZhBy?B�B�YB�)B D�B.;B	�	B!g�B�>B�*A���BMB�B�|B�Bz�B�9B��B#��BB�aB��B$YBR�B (�BZBN�B!��B�{B��Bm#B�%B#�BnVB�JBB�(B:�B:�B�B,b�B/B��B{ B,�B% �A�VB&@�B>�B�}B�]B
��BGCB�
B!�SBBxBA!B�hB·B��B >{B��B	�xB!8-B�PB�2A�w�B�OB��BŧB�AB��B�hB��B#YOB?�B��BDB$h8BAwB  �B>SB�-B!� B@mB��BB�B?�B#/{BAAB��B�;BXgB�-BH�B8)B,WcBBSB�kB�&A��A(XaA��"@��B�aA�ErAŪ�A�HUC�jA�+�@,A�A�,bA�WA�kVA���@f��A���A�A �iA��dA�-&A��A��A�@�A9OA��A�ȸA�Q0A��APV�@���AD��A�'A>��A�T$A:��A>heAt��A���?�C�AA;�@i;@{u@���Ai�@̸�A��A�g�Ar��A�X�C��A��C�XA�a<A<YA�uA(ԔA��@���B�eA�|�A�eA�FvC�g�A�e[@ʀA�tA�~UA���AjA�|�@i�A�q�A�u�@���A��A�~�A��RA�� A�~�A9qA��`A��mA�A�k�AP�@��cAE0�A؀�A>��A�e�A:ՒA>��As�À ?ʨ�A@�@h	6@| �@���AiK@͡"A�~�A�i�Ar�7A���C���A\C�IAÁA5               {   	      1   	                  8            �            #      	               L      =   "         %   
   !                        '   M      9   !      �      I                     O                     C         ;            )            %                     #      +            )                                 +      )   '                                 K                     3                                 %                     !                  )                                 )         %                  N��Ne|nN�a�On��P�MN7p�N��O��?NL�"N(�N��uP~NN��rNS�OmY�O�cN�ɳN���O���N,B�ON8��P��OJ��N��OM��O��NJ�ObXO�²O��,O�55O���N-�%O
�'P��N͂
Okj�N2n�N�iTN�O=<�Oy}O#�N;~EOX0�P��Na�O���O���N�A�O(�N+-O;<�N���N��g  B  �  ,    p  0  �  �  V  �  �  
  �    R    Z  @  U  �  �  �  �  �  �  �  �  �  �  
�  �  �    v  R  @  �  �  �  �  �  �    �  �  I  �  .  
  �  C  �  �  /  �  	&���u�D�����
<T��:�o;�o<u;�`B<t�<u=,1<u<D��='�<�o<�o<�C�=�Q�<�t�<�1<��
<�/<ě�<���<���<�/<�/<�/=H�9<��=D��<�<��=o=o=o=#�
=�P='�=��='�=49X='�='�=D��=P�`=q��=��P=�\)=�\)>o=���=�{=��=�
={z������������{{{{{{#*0220#"/4:70/"#05=GUbigaPI<0,%#����5Wehfl_NB5,��))26<BOQOBB6.)))))))��������������������cjt��������������tgcTUZanstnmaYUTTTTTTTTghknuz��zngggggggggg����������������������������"/5974)�������

���������)'-/<?HJHC</)))))))))057>BCBCB;5)'!)5B[glpqng[NB5) ��������������������KJUXannsxnaUKKKKKKKKLLO[gt��������tg[VPL��������������������lsz{�������������znl��������������������GEFDHTmz�������zaTIG��������������������_anz����znla________��������������������XTRUW[`hlrtvzz}tih[X����

�����������  #/<ACB9/#
���������������������������

�����~~�����������������~�����(--+)������������������������������
!###
	����������
#('
���������������������������������������������������������������������������Q[`hmt����{th[QQQQQQ#/3@IQRQHD</-'#�����������!).6BO[ahlmh^O@:6-)!����������������������� 
#/48:7.*#
	 �A??BO[hu}�����thOJFA"#/<B<8/#��������������������)5ANYipgN5����������������������������

��������������������������������

���������������������������������������������������������������������������������Ľнݽ��ݽٽнĽ������ĽĽĽĽĽĽĽ��T�a�h�m�p�q�m�a�T�H�F�>�H�K�T�T�T�T�T�T�r����������������������f�`�^�W�Y�f�k�rƳ������4�7�&�����ƳƁ�6� ��+�C�Y�}ƳÇÇÓÕÓÏÇÇÆ�z�z�o�zÆÇÇÇÇÇÇ�U�a�j�c�a�U�H�B�H�P�U�U�U�U�U�U�U�U�U�U�������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E����������������������������������������Ѻ����������������������������������������
�0�I�bŇŐřŖ�{�n�U�0���������������
�m�x�z�|�z�v�q�m�a�[�X�T�T�T�a�a�m�m�m�m������������������������������������������/�<�H�`�a�d�a�U�H�<�#�����������
����������������������������������������޻��!�-�7�8�:�<�:�-�!������������a�n�w�z�}�{�z�n�a�^�X�Y�a�a�a�a�a�a�a�a��)�6�H�Q�R�N�E�6�)�����������������ʼּ������ּʼƼ����������������������������������������������������������#�/�<�=�<�9�/�#���#�#�#�#�#�#�#�#�#�#�����
�#�0�=�A�;�2�#�
����������������������(�3�3�,�(�"���������ڿݿ���<�?�A�=�<�4�/�.�)�.�/�9�<�<�<�<�<�<�<�<�A�L�a�f�Z�U�M�A�6�(����
���%�(�3�A�"�/�;�H�T�a�m�o�m�a�\�T�H�;�/�!����"�5�A�N�Y�X�N�A�5�1�2�5�5�5�5�5�5�5�5�5�5�g�s�����������������������������y�s�]�g�����������������������ùëììù�Ҿʾ׾����������Ծ����������������ʻ������ûл׻ܻ޻ܻлû���������v�������f�t���������������s�f�d�Y�V�X�^�_�`�f�B�O�W�[�b�[�O�B�7�<�B�B�B�B�B�B�B�B�B�B�A�M�Z�f�r�s��}�s�f�Z�M�M�A�9�4�4�4�?�A�����)�B�[�r�o�h�]�W�N�B������������޾(�4�A�M�T�X�X�M�A�4�3�(�&�$�(�(�(�(�(�(�Z�f�s�t�|�}�y�s�f�]�Z�M�<�*�(�4�A�E�M�Z����������������������������������������ìù��������üùìàÖÚàâìììììì�@�E�L�Q�Y�_�`�Y�L�E�@�;�5�<�@�@�@�@�@�@�f�s�����������s�f�Z�M�C�A�=�?�A�M�X�f��!�-�6�:�=�=�:�-�)�!�������������-�:�F�K�O�S�Z�_�c�_�S�F�:�-�!����!�-���ûлܻԻлû��������������������������`�m�s�y��������y�m�`�T�G�;�3�;�E�G�T�`�'�4�M�f�r���z�r�Y�M�@�4���������'��������� �������������������������������;�T�a�m�|�������������z�m�a�T�H�.�%�0�;���������Ϳѿۿӿҿ��������y�o�i�n�{��������������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����������������y�t�s�y������������������EuE�E�E�E�E�E�E�E�E�E�E�E�E}EuErEpEoEpEu�<�H�U�U�\�a�b�a�U�H�<�4�/�+�/�0�<�<�<�<�ּ������������ּӼҼּּּּּ� U 7 % R ? O ^ @ # 1 ; H J ( Q X B D  � 5 8 + ; { R G 3 Y B # @ 0 B & \ 4 > O 8 v F + w `  C � H T >  V   1   �  u  �    �  w    .  ^  D  �  \  �  h  !  G    �  J  �  N  D  T  �  �  �  d  Z  (  �    I  ^  W  /  	  �  �  U  �  �  �    �  L  �  H  y  �  N  �  `  Z  �  �  �  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  B  >  9  5  0  )  "        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  t  h  U  B  /    	   �    	                +  &      �  �  =  �  �  ;  �      �  �  �  �  �  {  c  \  T  ~  �  ~  l  D    �  ]     	  Z  o  [  &  �  �  �  f  J  O  s  �    d  	  c    �  �  0  *  $          �  �  �  �  �  �  h  M  3    �  �  �  �  �  |  w  s  n  i  c  \  U  M  F  ?  7  ,  !         �  0  s  �  �  �  �  �  �  {  N    �  �  L  �  z  �  n  �    V  H  :  +    	  �  �  �  �  �  _  9    �  �  D  �  �  i  �  �  �  �  �  �  �  �  �  �  �  ~  z  u  q  l  h  c  ^  Z  �  �  �  �  �  �  �  �  �  i  D    �  �  j  !  �  1  �  \  �  	P  	�  	�  
  
  	�  	�  	�  	>  �  �  �  s  T  �    �  A  l  �  �  �  �  �  �  �  �  �  �  �  v  _  :    �  z  3   �   �    	    �  �  �  �  �  �  �  �  �  �  �  �  �    6  ^  �  '  �  �  �  �  �    /  C  P  R  0  �  �  ]  �  ^  �  9  �        �  �  �  �  �  z  P  !  �  �  �  F  �  �    �  �  Z  U  S  R  Q  Q  N  K  N  I  ;  "  �  �  ]    �  N  �  �  @  ?  >  =  ;  :  7  5  2  0  ,  '  #          �  �  �  �  �  �  �  4  �    C  U  <  �  �  .  v  q  +  �  	�  �  �  �  �  �  �  �  �  ~  r  b  M  8  #        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  h  M  #  �  �  �  Q   �   �  �  �  q  I    �  �  �  v  R  /    �  �  �  �  j  T  A  .  �  �  �  �  �  �  �  �  l  P  M  :     �  �  �  I    %    �  �  �  |  k  \  M  >  1  (      �  �  �  �  �  z  d  N  p  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  =    �  �  �  �  �  �  �  }  c  G  +    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  Z  =    �  �  �  �  X     �   i  �  �  �  �  �  �  �  �  �  }  k  X  A  )    �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  ]  D  %    �  �  �  |  �  �  	y  	�  
K  
�  
�  
�  
�  
�  
�  
H  	�  	�  	#  �    �  �  �  �    �  �  �  �  �  �  �  �  �  x  d  M  4    �  �  �  �  v  >    D  S  e  y  �  �    o  a  K  (  �  �  ?  �     ;  �   �    �  �  �  �  �  �  �  �  �  \  /  �  �  Q  �  �    �  �  v  ~  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  E  �  �  K  P  N  G  =  1  %    	  �  �  �  �  �  �  e  C    �  �  @  =  8  5  (  
  �  �  �  v  H    �  �  /  �  �  Y  �   �  �  �  �  �  �  �  z  w  u  n  b  U  F  6  %    �  �  �  q  |  �  �  �  �  �  �  `  ;    �  �  J  �  �  6  �  C  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  \  2    �  �  r  �  �    �   �  �  �  �  �  �  �  �  z  n  �  �  �  �  �  e  8  �  �      �  �  �  �  �  �  �  �  �  �  �  r  ?    �  �  F  �  �  |  �  	        �  �  �  �  �  �  f  B    �  �  1  �  V  �  �  �  �  u  b  [  b  p  o  b  N  5  "    �  �  Z  �  �    �  �  �  �  �  �  �  �  z  i  V  C  /    
  �  �  �  /  �    :  I  0    �  �  �  �  �  j  2  �  �  X  	  �  J  �  �  �  �  �  �  �  �  �  e  D  H  {  �  �  u  =  �  C  ~  �  ,  .  :  E  P  V  F  7  '    �  �  �  �  �  j  H    �  �  �  	g  	~  	�  	�  	�  	�  	h  	&  �  �  F  �  �  -  �  K  �  �  �  �  �  �  �  �  �  b  4    �  �  �  n  Q  '  �  �  j  �  K  �  C  ?  ;  8  4  0  ,  (  $             �   �   �   �   �   �   �  [  �  �  >  s  �  �  �  �  �  �  {    t  �  �    D  
Z    �  �  �  �  �  �  �  �  �  �  �  |  k  Z  H  6    �  �  �    )  .  '    	  �  c  �  v  �  T  �    
F  	[    �  }     �  s  K  #  �  �  �    U  '     �  �  �  �  �  �  U    �  	&  	  �  �  �  �  �  d  ,  �  �  |  @    �  �  G    �  �