CDF       
      obs    3   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�-V      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�0�   max       P�y      �  x   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �D��   max       >W
=      �  D   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?G�z�H   max       @E˅�Q�     �      effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�=p��
    max       @vF�Q�     �  (   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P`           h  0    effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�           �  0h   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �ě�   max       >l�D      �  14   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B,�      �  2    latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�gN   max       B+R�      �  2�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�C�   max       C��      �  3�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?���   max       C�V      �  4d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  50   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          I      �  5�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          3      �  6�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�0�   max       PAٟ      �  7�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��4m��9   max       ?لM:��      �  8`   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �D��   max       >W
=      �  9,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?G�z�H   max       @E˅�Q�     �  9�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�=p��
    max       @vF�Q�     �  A�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P`           h  I�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�$           �  JP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�      �  K   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Z�   max       ?ق����     �  K�                        �   5   a   	   $               $      R   
   
   <   6   =      )      '                  ;   $   ,                  (   	            
   :   -      N��]O��N��iORsN�?�O��1O�l�P��1P��P�>N���O|ՕNȉ�NS��NlN �O�i_N���PA$=O+k�O�hP�yP�PR?�N�'P�9OcR�O��sN]�M�0�N@��N�)�O�CdO��O�8�P<":NI1O�N���O2bN���O�N��N&@RO?AN��NF��O/bO%�N-2?N=�D���D��%   %   %   ;o;D��<t�<#�
<D��<D��<e`B<e`B<�o<�t�<�t�<���<��
<�1<�1<�9X<�9X<�9X<�j<���<�/<�`B<�`B<�`B<�h<�=+=\)=�P=#�
=#�
=,1=<j=aG�=e`B=ix�=q��=u=y�#=�O�=�^5=��=�l�>%>9X>W
=����������������������������������������XX[]fgt~�}wtg[XXXXXXtmmqtz�����������ttt��������������������-5[b`ft����gNB(bc|��������������thb5Bg�������gRB*#HUavz����znWUH<#�����$5NUTL@5!�������
 
�����������������98<BO[chihc[VOKB9999/-07<IUXUOIA<0//////`anz����zona````````��������������������������!"!#' ���D?HOUanrz}�zynaWULHD���������� ���������������������������__amz����������|zma_��5Ngx������t[B5)�	
#/>UakxzaU/#	���)3B[cpdWB)��������������������������6BOW[XOD6���!%/25=?NTRRJKHB5)����������������������������������������~|y���������~~~~~~~~)6;>:6-)����������������������� ���������������������������������suuz��������������zs������)1795)����88;=GHIJKHD;88888888#&-/34984/-'#mjkst��������tmmmmmmA<;>HTadgecaYTLHAAAA	()**)( �������)5<BB95���0230%# 
	
	
#(0**0))**********#,/277///#
	
#��������������������#-*%#��������������������:64359<HUU\_`]VUH<::wsvz~�������zwwwwwwzzz������zzzzzzzzzzz�{�}ŇňŉŇ�{�n�b�`�b�g�n�v�{�{�{�{�{�{�����������������������������~�y�z�����������������������������������������������zÇÓàåìòíìàÚÓÇÅ�z�w�r�z�z�z�*�6�C�I�C�?�<�6�*�'��������'�*�*�G�W�m�y���{�m�`�T�N�L�G�;�.�#�$�*�.�;�G���(�A�N�O�K�A�5�7�4�(���������āĚĦįģĠĔā�h�B�6�)�$�%���)�B�Oā�������)�5�9�)�����������������Ŀ��U�nŇřŠ�{�b�U�<�#�
����ĿıķĸĿ�������(�/�(�%������������������ܻ������)������û����ûȻлջܿѿݿ�������ݿڿѿƿĿÿĿɿѿѿѿѼ����������������������{�~�������<�@�B�@�=�<�2�/�,�/�0�9�<�<�<�<�<�<�<�<ÇÓÔàåàÓÇÀ�|ÇÇÇÇÇÇÇÇÇÇ�:�F�S�_�l�x�����x�l�S�F�:�-�!����)�:�zÇÎÇÃÆÉÇÁ�z�n�c�a�^�\�a�c�n�s�zàìù����������������ì�z�k�\�`�rÉÓà�G�T�`�m�y�������y�m�`�T�G�;�6�6�;�C�B�G�ѿݿ�����ܿѿĿ����������Ŀǿ˿οѿ���5�T�m�~�����������@������Ͽ����꾥���ʾ׾���׾ž��������~��z��������ƎƧ���������������ƧƎƁ�s�_�`�k�uƎ�/�<�H�R�T�H�E�=�<�0�/�/�#�"� � �#�%�/�/������������������ʾ��������������	��"�/�;�H�M�H�;�7�/�"��	�����������	�s�w�v�s�u�z�s�f�Z�A�$�����(�4�A�c�s���'�'�-�'���������������:�F�S�X�S�F�>�:�-�!�-�3�:�:�:�:�:�:�:�:ÓÕ×ØÓÇ�z�s�z�}ÇÈÓÓÓÓÓÓÓÓ�5�A�N�Z�]�g�s�t�s�h�g�Z�N�A�=�5�)�3�5�5�ǽ½������������}�u�������������ƽƽнǼ����ͼּ����ּʼ���������n�l�r����Šŭ������������������ŭŠŔŌŇŇŗŘŠ���	��"�/�;�=�:�+�"��	���������������������	�
����
������������������������������������������������������������v������������������߻߻�������U�b�n�{ŅŅ��{�n�b�U�P�I�C�I�K�U�U�U�U�Y�e�r�~���������������~�w�r�e�[�Y�X�Y�Y�������
������������������������侤��������~������������������þ��������?�3�'���'�3�@�C�A�?�?�?�?�?�?�?�?�?�?ǭǡǜǔǈ�{�o�o�o�r�{�ǈǒǔǡǭǰǲǭ�����������������������������������������*�6�C�O�N�C�6�*�(�)�*�*�*�*�*�*�*�*�*�*D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�Ź�������������������������ŹŷŭųŹŹ������!�.�9�.�!��������������������4�@�E�L�@�4�'�&�'�4�4�4�4�4�4�4�4�4�4�4 ; " A  F W q : u J + A ) _  Q 8 Y " , { f I 7 T D A O   � P Q L F ' 0 m { # ( H b C < S 6 - 5 8 } 9    �  ,  �    �  d  �  Y  �    �    �  �  �  )  2  5  '  {  �  �  r  �  !  �  �  s  e  7  q  �  �  Y  �  C  d  �  �     �    	  D  C  �  _  u  i  �  1�ě�;ě�<T��<�/;�o<�j<�1>H�9=�%=�/<�1=L��<���<�9X<�j<�j=]/<�/=��`<�<��=��=���=���='�=�C�=49X=�+=\)<��=t�=D��=e`B=ě�=���=�1=<j=m�h=�%=��=�%=ȴ9=�C�=�+=��T=���=�l�>/�>.{>I�^>l�DBftB"�{B	d�B
\�Bl�B�%B �B	vB��B�B$[Bo$BǈB&p�B��B!�B�LBBB�FB �3B �PB�AB�B�B�vB�B_7BB7B#�B\B[�BثB,�B��BS6B$A��B��B�SA�RB��B�}B$��B�B�BB��B?yB9rB_B�B��B"�sB	A�B
BoB��BE�B >�B�$B�6B�B$~�B@B�B&��B��B"AB��B��B��B E�B<2B��BA�B�ABЯBo�B?-B@�B#@FB��B��B��B+R�B�{B��BZLA�gNB�GB��A�~B�B��B$�B$�B�B7DB@LB?�B>�B?�B��A��@��IA���A�^eA���Ae��A6� AکAҖ$A��A2�@�M�A|�@�w9Aê�A�.f@���A�?�A��TAh�pAy�=A� AL�RB"CA�AR-A�~�A=ϭ?�C�@��Aɾ�A���A u@�x1A�1�A��vA��/A��g@�*A��v@ 2A�W|AJj<?��)B�A��hB q�C��A�ZA�@��9A�~F@��A��kA�g�B J�Ae��A8�|AڀDA�~sA���A1	$@���A{{@���Aô�A�nq@���A�/A���Ai�Ax�PA���AL�/B(8A��ARޕA�p�A=J�?���@��Aɀ8A���A"��@��jA�{�A���A�8A��@�A�H�?���A�r�AK�{?��B"�A��MB �*C�VA�pA	i@�
�                        �   6   a   
   %               $      R   
   
   =   6   =      *      '                  <   $   -         	         (   	               ;   -                        %   '   5   -   ;                           +         I   '   1      '                     "   '      +                                             
                              '                           !         3   %   '      %                     "         +                                             
N��]N���N��iN�*=N�?�O�W@N�vO���N��<PW(N��O{�Nȉ�NS��NlN �O00N���O��N=��N�6xPAٟO��P��Nau�O��NO>��OFeN]�M�0�N@��N��O�CdNуRO�8�P<":NI1N�� N���O2bN���O+��N��N&@RO?AN��NF��O/bO>N-2?N=     �    �    V  �  �  �  {  �  {  �  �  m  �  >  3  	�  �  >  �  x  �  �  �  $    �  �  �  �  �  d    �    1  �  1  2  ^  L     �    �  �  	�  R  l�D���ě�%   ;�o%   <o<49X=��=��=49X<T��<���<e`B<�o<�t�<�t�<�`B<��
=D��<���<ě�=�w<ě�=�P<��=o<�=C�<�`B<�h<�=C�=\)=�O�=#�
=#�
=,1=D��=aG�=e`B=ix�=�\)=u=y�#=�O�=�^5=��=�l�>1'>9X>W
=����������������������������������������XX[]fgt~�}wtg[XXXXXXoost}���������ytoooo��������������������)5BNXW[gwtf[NB)"��������������������338CN[gw�����tgNB:53-)'/<HNUVUKH</------�����):DF?5)�����

 ������������������98<BO[chihc[VOKB9999/-07<IUXUOIA<0//////`anz����zona````````����������������������������D?HOUanrz}�zynaWULHD����������������������������������������vz�����������zvvvvvv)BNg�������[B5)#/=U_jwynaU/#

5BN[b_YL5)�������������������������6BRXUOB6��%#'25;BNQPQIIFB5,%����������������������������������������~|y���������~~~~~~~~)6;>:6-)����������������������� �����������������������������������suuz��������������zs������)1795)����88;=GHIJKHD;88888888!#-/0652/+&#mjkst��������tmmmmmmA<;>HTadgecaYTLHAAAA	()**)( �����������0230%# 
	
	
#(0**0))**********#,/277///#
	
#��������������������#-*%#��������������������8557;<HOUX\\YUPH><88wsvz~�������zwwwwwwzzz������zzzzzzzzzzz�{�}ŇňŉŇ�{�n�b�`�b�g�n�v�{�{�{�{�{�{��������������������������������������������������������������������������������ÇÓàâìïìéàÓÇ�{�z�u�z�~ÇÇÇÇ�*�6�C�I�C�?�<�6�*�'��������'�*�*�G�`�n�u�{�u�m�`�S�J�G�;�.�)�)�/�6�;�B�G���$�(�4�A�E�G�A�@�4�4�(��������[�h�tāćĈćā�t�h�[�O�B�;�8�8�<�B�O�[������
���������������������������������#�<�I�c�f�`�W�I�<�#�
������������������'�!��������������������������������ܻڻлͻлһܻݻ�ѿݿ�������ݿڿѿƿĿÿĿɿѿѿѿѼ����������������������{�~�������<�@�B�@�=�<�2�/�,�/�0�9�<�<�<�<�<�<�<�<ÇÓÔàåàÓÇÀ�|ÇÇÇÇÇÇÇÇÇÇ�-�:�F�S�_�k�x�z�x�s�l�_�S�F�:�-�(�#�*�-�zÇÎÇÃÆÉÇÁ�z�n�c�a�^�\�a�c�n�s�zì����������������ùàÎÇ�z�t�v�zÁÓì�T�`�m�s�m�l�b�`�_�T�K�R�T�T�T�T�T�T�T�T�ѿԿݿֿܿѿĿ��������¿ĿϿѿѿѿѿѿѿ����(�H�_�e�k�s�m�m�g�N�A�+���ܿؿ꾱�ʾ׾���׾ľ��������������|��������Ƨ����������������ƳƚƎƁ�r�k�p�~ƎƚƧ�/�<�H�H�J�H�A�<�/�&�&�-�/�/�/�/�/�/�/�/�������׾�����	�
����ʾ������������������	��"�/�;�F�;�7�3�/�"��	�������������Z�f�r�s�o�q�u�s�f�Z�L�A�-�(�$�(�4�A�H�Z���'�'�-�'���������������:�F�S�X�S�F�>�:�-�!�-�3�:�:�:�:�:�:�:�:ÓÕ×ØÓÇ�z�s�z�}ÇÈÓÓÓÓÓÓÓÓ�N�Z�Z�g�q�g�e�Z�N�A�?�5�+�5�A�B�N�N�N�N�ǽ½������������}�u�������������ƽƽнǼ���������������������������������������Šŭ������������������ŭŠŔŌŇŇŗŘŠ���	��"�/�;�=�:�+�"��	���������������������	�
����
�������������������������������������������������������������z������������������߻߻�������U�b�n�{ŅŅ��{�n�b�U�P�I�C�I�K�U�U�U�U�Y�e�r�~���������������~�w�r�e�[�Y�X�Y�Y�����
������
���������������������񾤾�������~������������������þ��������?�3�'���'�3�@�C�A�?�?�?�?�?�?�?�?�?�?ǭǡǜǔǈ�{�o�o�o�r�{�ǈǒǔǡǭǰǲǭ�����������������������������������������*�6�C�O�N�C�6�*�(�)�*�*�*�*�*�*�*�*�*�*D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D���������������������������žŹųŹź���Ƽ�����!�.�9�.�!��������������������4�@�E�L�@�4�'�&�'�4�4�4�4�4�4�4�4�4�4�4 ;  A  F G X 3 1 0 % % ) _  Q 4 Y 4 < _ e J # ? : 7 5   � P M L  ' 0 m u # ( H 9 C < S 6 - 5 5 } 9    �  �  �  �  �  K  ,  �  �  �  �  )  �  �  �  )  �  5  �  [  �     f  m  v  1  �  �  e  7  q  �  �  �  �  C  d  ;  �     �  �  	  D  C  �  _  u  #  �  1  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  k  [  [  o  {  �  �  �  �  �  �  �  u  X  2  	  �  �  ]  (  �  �        �  �  �  �  �  �  �  �  �  o  %  �  �  q  1  �  �  �  �  �  �  �  �  �  �  �  h  F    �  �  R  �  [  �  �  �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    3  S  O  L  K  M  S  U  J  :  %    �  �  �  �  b  9    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  g  %  �  Z  �  �  x  �  \  �  !  h  �  �  �  }  5  �    L  ,  
�  $  �  �  �  �  f  �  �  �  J  {  �  �  �  �  �  4  �  $  v  �  	  �  n  �  
  N  s  z  z  r  Y  ,  �  �  �  e  �  q  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  e  E  %    �  �  �  �  k  �    2  `  s  z  z  j  G    �  �  |  0  �  X  �  D  �  �  �  �  �  �  �  �  �  �  �  �  �  |  l  _  U  O  K  E  =  3  �  �  �  �  �  |  s  l  f  `  `  h  p  s  l  f  ]  F  0    m  g  b  \  V  N  E  <  2  %      �  �  �  z  H    �  �  �  �  �  �  �  �  �  �  �  �  }  q  h  a  Y  R  >  '    �  �  �    ,  7  >  :  /  !    �  �  �  �  N  �  �  J  �  Y  3  '         �  �  �  �  �  �  �  x  R  E  W  i  r  w  |  �  l  �  	6  	�  	�  	�  	�  	�  	�  	A  �  �    �    �  �  �  4  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        *  ;  <  8  .  !      �  �  �  �  �  �  �  w  \  $  R  J  T  �  �  �  ]  O    �  �  &  S  6  �  }  �  �  �  l  w  p  g  X  A  %    �  �  �  Y  �  �    �  �  @  �  �  w  �  �  �  �  �  �  �  ~  [  3    �  �  ;  �  =  �  �  p  �  $  T  |  �  �  �  �  �  �  x  d  O  7    �  �  �    8  �  �  �  �  �  �  �  �  �  �  �  �  \  %  �  �  9  �  C  b    !  $  !         �  �  �  q  C    �  �  �  Q    �  �  >  �  �    �  �  �  �  �  x  L  $  �  �  �  ?  �  �  9  �  �  �  �  �  �  �  v  k  `  W  O  F  >  6  -  $      	     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  �  �  �  �  �  �  �  �  �  |  m  ^  N  %  �  l     �   �   �  �  �  �  �  �  �  x  U  /    �  �  _    �  �  5  �  r  	  �  �  �  �  �  �  �  �  �  s  O  !  �  �  T    �  �  �  �  V  P  }  �  �  �  �    8  N  _  P    �  m    �  �  �  �    �  �  �  �  �  �  �  �  {  d  E    �  �  B  �  ^  �  �  �  �  �  �  y  a  2  "  �  �  �  �  d  /  �  �    �  5  \          $  )  /  4  :  ?  E  I  N  S  W  v  �  �  �    +  )  *  0  -  $       �  �  �  �  �  �  �  �  t  C  �  �  �  �  �  v  c  P  =  *    �  �  �  �  �  i  \  U  z  �  Y  1  .  )  #        �  �  �  �  x  ;  �  �  S  �  �  @  �  2      �  �  �  q  K  '    �  �  �  �  �  �  �  �  �  �    �  �  �    Z  Z  E  '  �  �  �  L  �  �  C  �  G  _    L  #  �  �  �  �  �  �  t  c  R  C  3       �  �  �  �  �       ,  B  N  5      �  �  �  �  b  @    �  �  �  �  �  �  �  �  w  W  7    �  �  �  �  ~  W  +  �  �  f  4    �            �  �  �  �  z  K    �  �  n  /  �  �  �  �  �  �  �  �  �  �  n  R  6    �  �  �  �  �  �  �    '  7  �  ^  1  �  �  �  =  �  �  A  �  Q  
�  	�  �  �  u    �  d  	�  	�  	�  	�  	�  	�  	�  	t  	P  	%  �  �  _  �  �  �  �  v  (  �  R      �  �  �  �  X    �  :  �  c  �  v  �  `  �  <   �  l    �  �  �  �  �  �  9  
�  	�  	8  �  �  �     K  |  �  �