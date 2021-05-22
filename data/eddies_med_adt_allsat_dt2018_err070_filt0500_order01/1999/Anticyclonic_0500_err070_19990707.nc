CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?ȴ9XbN      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P�f�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �+   max       =���      �  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @E������     �   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���R    max       @vw
=p��     �  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @N�           p  1�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ˊ        max       @�f�          �  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       >hr�      �  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�"�   max       B.�8      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�    max       B.��      �  4�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��   max       C�mW      �  5�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�B�   max       C�x      �  6�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3      �  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P~��      �  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�Q   max       ?���>B[      �  :�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �+   max       =�F      �  ;�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�p��
>   max       @E������     �  <�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���R    max       @vvz�G�     �  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @N�           p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ˊ        max       @�u@          �  N�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�      �  O�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�L�_��   max       ?���>B[     �  Pl               	                                             )   l      U            	         (   ;   �         	                  A         .   <   /      '         )         a   	   cN�A.M��DN�eN��eN�ON�7�N.o~O�8+N)��O5eO�<QNޢ�N��N��OH9�O9��NGCcN�r�Oh�P9P�f�N��dP�-O�oN[��N k�O-�OָOo'Oځ|O���PJ�ON1x5N{؂N�KOQ^�N1��Oy��N/��O��O� qOb��O��yPq�NP�Q�P��O��9O�:�O�M�Of�O��
M��OY4eP�@N��O:*`�+����1���
�49X�ě��ě��ě����
�o%   %   ;��
<t�<t�<T��<e`B<�o<�t�<���<��
<��
<�9X<�j<���<���<�/<�/<�<�=o=\)=�P=�P=��=��=��=��=��=�w=#�
=#�
='�=,1=,1=,1=0 �=0 �=D��=Y�=ix�=m�h=y�#=}�=�7L=���--./<HQTQIH<2/------ropt����wtrrrrrrrrrr
 )*.35865)
������������������������������������������� �����������������������./5<GIITblruvunbI<1.#0<IKIC<0(#ZZXXabnz�������znbaZ�
/;HIHGFGF>;/"	#*/6<?CB@</-&###/<<<941/#��������������������41533<IbnrnmnoibXU<4lhenpz������������zlzy~�����������zzzzzz���)*+*�������������������������eegt�������������mge�������	)NXUQB����{x��������������{{{{+#%.<HUa��������aU<+�����#254+#
������������������������������

�������������������������������������



	���������������$)BO[ddVSROC6'�����#/3.)!
������:7;BN[gw��������gOB:��������������������^gktx�����{tjg^^^^^^�|~���������������������)-//,)��20/.59>BCBB522222222���������

���.//<EHKHH<;/........#/<FHLOH@</#�����
#'($
���������
#/;<D@</#
��
:F[gt��ztN5)xstw���������������x������-5B[t|[V6���������	
���������������������������������)5BN[ff[N)�#*4<?FHHA</-#dbjnr|�����������znd)-B[hwxukhb[OD@82)*+3)')**********;;;BHQTakpqmjaTHA><;��������������������������������������������

�����EEEEE#E#EEED�D�D�D�D�EEEEEE�n�zÇÒËÇ�z�v�n�l�n�n�n�n�n�n�n�n�n�n���ʾ׾�������׾ʾľ��������������������������������������������������B�O�[�_�[�P�O�B�7�A�B�B�B�B�B�B�B�B�B�B�	��"�/�3�/�-�"��	������������������	�����������������������������������������������ɻлֻлû����������x�j�g�t�y�����4�5�:�;�<�4�0�'�!�"�'�2�4�4�4�4�4�4�4�4����������������������������������������H�T�a�z�����m�a�T�H�;�/�"� ���*�/�@�H�5�A�N�Z�g�Z�W�N�F�A�5�(�����(�.�5�5������������������������������������������������ ��������ܻ����������������������������������y�r�p�r����������������������������������������˺@�L�Y�[�e�f�n�e�Y�W�L�L�@�>�@�@�@�@�@�@�������������������������������������������������ûȻλлһͻû������������������/�H�Y�d�e�d�_�H�/�"��	������������"�/������UŇŎŊ�{�I�#�
������ļĵĸ������G�T�V�U�T�K�G�;�3�.�)�%�.�3�;�B�G�G�G�GE�E�E�E�FFFFE�E�E�E�E�E�E�E�E�E�E�E��a�z���������������z�a�P�H�6�9�D�H�Q�T�a�����	��	���������������𾥾��������������������������������������4�A�G�M�M�I�A�5�4�(���������)�4�A�K�Z�f�s�����s�k�f�_�Z�Q�M�:�8�3�4�?�A������������������������{��������������-�:�G�S�U�S�V�P�F�:�!�����ֺݺ���C�N�[�a�b�d�d�`�N�5�)������ ��)�5�C��)�B�Q�`�e�c�[�O�8�����������������������������������������������������������u�~ƁƄƁ�{�u�h�\�Z�\�_�h�n�u�u�u�u�u�u������	����
������������������������H�U�a�f�n�n�r�n�m�b�U�H�<�/�&�,�,�8�B�H�����
����
��������������������������H�U�a�n�z�|ÇÓàäÓÇ�z�a�U�<�7�4�<�H¤�|�������������������������������EuE�E�E�E�E�E�E�E�E�E�EuEiEbE_EcEiElEuEu�`�m�y�������������y�x�`�T�Q�Q�P�R�T�Y�`�(�A�M�Z�_�Z�M�D�N�P�M�4�(�%����(�$�(�g�����������������������N�5�+�,�0�C�P�g��������(�/�-��	� ������ݿ��y�`�Y�m�������Žɽƽ��������y�`�S�E�C�G�S�`�l�����T�a�m�����������������m�a�T�;�+�(�2�;�T�����������������������������������������������ľ�����������s�f�M�A�D�M�_�s������(�4�8�A�M�Z�f�d�X�M�A�4�(�$��������ļʼӼ˼Ƽ�������r�f�Y�N�Y����������?�3�'� �$�'�3�6�@�B�?�?�?�?�?�?�?�?�?�?�����������������������ĿıĮĨĳĿ���弘�����ļ������ּʼ����������r�f�e�r��ǭǡǔǈ�{�v�r�{ǈǔǡǭǰǭǭǭǭǭǭǭD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�   P G Y m P v 9 { % 9 Z N ` ' , v d 1 & 2 k N 8 + e 7 h ' 5 , ) d M 2 , o S : C 0 ; y @ H ( T J r F K L 3 m D =  �  /  /  �  "  /  �  5  �  }  t    �  �  �  �  �  �  �  A  }  �  �  !  f  Q  �  _  9  �    f  �  �  �  �  d    I  9  P  �  .  �  �  I  �  �  �  �  �    �  @  �  ���j���ͼT����C��o��o��o;�`B��o<#�
<ě�;�`B<49X<�1<�j<ě�<���<��
=<j=u>J<�`B=�"�=8Q�<�`B<�=t�=�P=,1=�O�=�Q�>hr�=,1=49X=@�=m�h=0 �=�%=L��=ix�=���=y�#=P�`=�9X=��=�E�=�O�=��=�t�=�+=Ƨ�=�%=��
>!��=���>:^5B�B	��Bu�B��Bv�B�FB!�aB'%�B%��B�SA�"�BE�B��B"�B&�SB�[B mB.�8B!n�B
ؔB;�B
��B?ZB�B Z�B#�HB X$B#��BʡB�B;dB	U�B��B	�aB ��B��BOIB�B�%B[�B��B|�B�B�]B�oB,��B��B/B��B�B��B&mA�>4B Br<B#�B�NB	��B6�B��B@~BɮB!�CB'�B&8UB�NA� B�B�wB">B'<�B�B7,B.��B!H�B
�B��B<�B�oB�B 8gB#B.BйB#��B�nB�|B>B	?�B@�B	��B6�B��B1QBF�B�;B��B�BAVBj�B�B?zB,�XB@�B?�B?�B��B�"B;1A�~�B?�B��B?nC�b�A��7AQx�A�K�A�[A���@# @���@�'�A�
A�ZA��A��,@�F@쥊A��?�i�AqK�@�*�A�LQA�D�Ac�C�mWA��AW��AK
�A87�A?�AI�Z@kMZA���A��xAu�BxGA�P.A�ʹA��MA�>}A��OA�!�C��Aj�9A:��A���As��A��A�J�A�~�AE:dA9�@���?��A�K�@�҂B4�C��C�a.AȢAR.A���AفA���@$�@��W@��A�c�A��A���A���@���@��A�p!?��Ar��@���A��A��Ab��C�xA�}�AX�GAKi�A8�FA?-AI�@g��A���Aԃ�AsPB�0AўiAņ�A��AƄ�A��A���C���AkC9A9A�A���Ar�A��A���A�~�AD��A9+�@���?�B�A�֮@���B��C���         	      
                                             *   l   	   V            
         (   ;   �         
                  A         /   <   0      (         )         a   	   d                                                            %   =      %   #                  %   #   +                                 '   1   =   )   !            #         /                                                                     3      !                                                            '   1                     !               N�A.M��DN�eN��eN�ON�7�N.o~O�8+N)��O��O�UKNޢ�N��N��OIzO9��NGCcN�r�Oh�O�ǐP~��N��dO�JoO���N[��N k�O��Nw��Oo'O���O�l�O�}�N1x5N{؂N�KOQ^�N1��Og�N/��N�oOjq�Ob��O��yPq�NO��O?^�O���O�h�O�M�Of�O��M��OY4eO-V1N��Ox\  E  �  �  �  N  b  �  �  �    0  �  y  �  q  �  r  �    �  	Y  �  	  �  a    /  �  ,       ?  ~  �  %  �  k    A  8  
4  5  �  �  �  �  �  8  L  �      v  �  ?  [�+����1���
�49X�ě��ě��ě����
%   ;o%   ;��
<t�<D��<T��<e`B<�o<�t�<�h=8Q�<��
=��<�h<���<���<�`B<�<�=t�=D��=�F=�P=�P=��=��=��=�w=��=,1=D��=#�
='�=,1=�\)=}�=<j=D��=D��=Y�=u=m�h=y�#=�/=�7L=���--./<HQTQIH<2/------ropt����wtrrrrrrrrrr
 )*.35865)
������������������������������������������� �����������������������./5<GIITblruvunbI<1.#0<IKIC<0(#^]Z[agnz|������znla^	/6;AHGEEFD;/"	#*/6<?CB@</-&###/<<<941/#��������������������85;8<@IUXbjiibbUI<88lhenpz������������zlzy~�����������zzzzzz���)*+*�������������������������rljjlot������������r��������;GHA4���{x��������������{{{{/**-3<HUau������aH</�������#(/00*#
����������������������������

����������������������������������������


��������������)BO[`XRPPOJB6)�������
#%#
�����KHGIN[gt�������tg[RK��������������������^gktx�����{tjg^^^^^^�|~���������������������)-//,)��20/.59>BCBB522222222������ 

����.//<EHKHH<;/........!#/<BHIMH=</+#������
 $%"
��������
#/;<D@</#
��
:F[gt��ztN5)xstw���������������x���)5:BDFB>5)�����������������������������������������)5BN[bc[NB)#*4<?FHHA</-#dbjnr|�����������znd3,)1BZhuvtih`[OJFA93*+3)')**********;;;BHQTakpqmjaTHA><;�����������������������������������������������

������EEEEE#E#EEED�D�D�D�D�EEEEEE�n�zÇÒËÇ�z�v�n�l�n�n�n�n�n�n�n�n�n�n���ʾ׾�������׾ʾľ��������������������������������������������������B�O�[�_�[�P�O�B�7�A�B�B�B�B�B�B�B�B�B�B�	��"�/�3�/�-�"��	������������������	�����������������������������������������������ɻлֻлû����������x�j�g�t�y�����4�5�:�;�<�4�0�'�!�"�'�2�4�4�4�4�4�4�4�4�����������������������������������������H�T�[�a�x�z�}�z�p�a�T�H�;�/�!��"�/�E�H�5�A�N�Z�g�Z�W�N�F�A�5�(�����(�.�5�5������������������������������������������������ ��������ܻ���������������������������������v�v�x�����������������������������������������˺@�L�Y�[�e�f�n�e�Y�W�L�L�@�>�@�@�@�@�@�@�������������������������������������������������ûȻλлһͻû�������������������"�/�;�H�T�Y�\�Z�T�H�/�"��	�����������
�0�I�j�{�~�y�p�H�#�
����������������G�T�V�U�T�K�G�;�3�.�)�%�.�3�;�B�G�G�G�GE�E�E�E�E�FFFFE�E�E�E�E�E�E�E�E�E�E��m�z���������������������z�m�^�S�U�W�b�m�����	��	���������������𾥾��������������������������������������4�A�B�K�K�F�A�<�4�(��������(�-�4�A�M�Y�Z�\�f�h�s�y�y�s�p�f�Z�M�C�A�@�A�A������������������������{�������������-�:�E�O�P�O�F�:�-�!�����������-�5�B�N�W�[�]�[�[�[�O�B�5�)���
���)�5����)�9�F�G�@�0�)���������������������������������������������������������u�~ƁƄƁ�{�u�h�\�Z�\�_�h�n�u�u�u�u�u�u������	����
������������������������H�U�a�f�n�n�r�n�m�b�U�H�<�/�&�,�,�8�B�H�����
����
��������������������������U�a�n�zÇÓàãÓÇ�z�a�U�H�<�7�5�<�H�U¤�|��������������������������������EuE�E�E�E�E�E�E�E�E�E�E�EuEiEeEbEgEiEsEu�`�m�y�������������y�x�`�T�Q�Q�P�R�T�Y�`�(�A�M�Z�_�Z�M�D�N�P�M�4�(�%����(�$�(�g�����������������������N�5�+�,�0�C�P�g���������ȿοҿӿѿɿĿ������x�r�n�u�����y���������������������y�l�_�W�S�S�`�h�y�a�m���������������z�m�a�T�C�;�1�,�7�H�a�����������������������������������������������ľ�����������s�f�M�A�D�M�_�s������(�4�8�A�M�Z�f�d�X�M�A�4�(�$������������ʼѼʼļ�������y�f�^�U�Y�f������?�3�'� �$�'�3�6�@�B�?�?�?�?�?�?�?�?�?�?�����������������������ĿıĮĨĳĿ���弱���ʼּ�����ڼּʼ���������������ǭǡǔǈ�{�v�r�{ǈǔǡǭǰǭǭǭǭǭǭǭD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�   P G Y m P v 9 { $ 3 Z N ` " , v d 1 $ 7 k P ? + e 4 e ' , ! " d M 2 , o T : < " ; y @ %  O G r F G L 3 @ D (  �  /  /  �  "  /  �  5  �  >  8    �  �  8  �  �  �  �  {  :  �  7  7  f  Q  L  �  9  s  /  8  �  �  �  �  d  �  I  �  �  �  .  �  |  �  �  Z  �  �  �    �  |  �     F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  E  6  &      �  �  �  �  y  U  /    �  �  r  ;    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    w  n  f  �  �  �  �  �  �    l  U  =  "    �  �  �    L     �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    ,  N  B  6     	  �  �  �  �  �  �  {  S  +    �  �  {  J    b  _  \  Y  U  L  B  9  0  '          !  &           �  �  �  �  �  �  �  w  m  f  ^  W  R  P  N  L  ?  /      �  �  �  �  �  �  �  y  h  U  ?  %    �  �  �  }  [  9    �  �  �  �        	  	  	         �  �  �  �  �  �  �  �  �       �  �  �  �  �  �  �  �  �  l  J  &    �  �  @    /  /  '      �  �  �  �  k  C    �  q  	  �    w  �  �  �  �  �  �  �  �  �  |  p  d  X  E  2    �  �  �  �  c  y  q  i  a  W  M  C  8  -  "    !  $      �  �  �  �  �  �  �  �  �  o  i  �  �  �  �  �  �  �  �  {  j  Y  F  3    `  f  k  o  q  q  m  g  \  M  :  #    �  �  �  �  w  C   �  �  �  �  �  o  Y  A  )      �  �  �  �  �  �  d  2   �   �  r  m  i  d  ]  P  C  5  #    �  �  �  �  �  �  �  f  G  (  �  �    v  m  c  T  E  6  '       �   �   �   �   �   �   }   h    �  �  �  �  �  m  N  *    �  �  V    �  �  c  '  �  �  �  �  �  �  �  �  �  �  �  �  ]  4    �  �  =  �  !  v  D  �  �  	(  	H  	W  	W  	E  	  �  �  �  1  �  M  �  p  �  �  �  �  �  �  }  n  Z  F  4  !      �  �  �      "  %  (  +  /  
�  
�  
�      
�  
�  
K  
  
K  
>  	�  	�  	  j  �  �  .  1   y  �  �  �  �  �  �  �  �  �  �  �  �  b  8    �  �  W  �  Z  a  ]  Y  V  R  N  J  F  B  >  9  2  ,  %        
     �    �  �  �  �  �  �  �  �  �  �  �  �  }  x  y  }  �  �  �  -  .  /  +  &        �  �  �  �  �  �  �  d  *   �   �   �  �  �  �  �  �  �  �  �  �  h  E  6       �  �  �  a  3    ,  %          �  �  �  �  �  �  �  �  �  U    �  �  g  �  �           �  �  �  �  R    �  ]    �  i    �  �  �  �  �  �  �     �  �  �  �  M    �  i  �  b  �    4  �  �    �  �  �  k  �    =  1    �  R  �  �  @  �  �  m  �  ~  x  r  l  f  [  Q  G  <  0  $    
  �  �  �  �  �  Z  -  �  �  �  x  m  c  Q  :  $    �  �  �  �  �  z  a    �  N  %  
  �  �  �  �  |  _  C  *       �  �  �  �  �  �  �  �  �  z  c  N  5  "    �  �  �  �  y  J    �  �  \    �  '  k  W  C  .      �  �  �  �  o  M  )    �  �  �  i  ?    �    �  �  �  �  �  �  �  �  �  �  w  U  -    �  �  �  �  A  +      �  �  �  �  �  �  �  v  _  H  3    
  �  �  �    /  5  8  1  #    �  �  �  N    �  �  O  	  �  x  )  �  	�  

  
1  
0  
  
   	�  	�  	�  	I  	  �  �  o  	  x  �  �  K  �  5      �  �  �    M    �  �  a  $  �  �  e  #  �  �  4  �  �  �  �  �  u  [  =  $      �  �  �  �  �  o  <     �  �  �  �  �  �  ~  X  /    �  �  c    �  �  k  $  �  F    o  f  Q  A  k  �  �  �  �  �  �  �  }  K    �    \  �  �  g  z  �  �  �  �  �  �  �  �  �  �  k  W  9     �  '  O   �  �  �  �  �  �  �  �  �  e  J  3  )  	  �  �  m  )  �  �  g    5  7  6  ,    �  �  �  f  .  �  �  p    �  �  t    �  L  G  <  '    �  �  �  �  �  r  G  &  �  �  2  �  Q    �  �  �  �  �  z  h  U  @  +    �  �  �  �  �  z  R    �  �  �  �  �  �  �  �  R  +    �  �  �  �  U    �  �  >  �  �    
      
  '  C  `  v  �  �  �  �  �  z  n  ,  �  �  <  v  r  g  W  B  #    �  �  �  v  M  "  �  �    H    �  �  �    �  �  �  ~  l  �  ^  �  �  K  �  o  
�  	�  �  R      ?  "    �  �  �  �  �  n  V  >  %    �  �  �  �  �  j  C  �  R  :  [  Q  =    �  �  6  �  )  o  �  �  �  �  
�  h  �