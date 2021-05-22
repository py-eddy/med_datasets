CDF       
      obs    6   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?��x���      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�Y   max       P��~      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       >�+      �  \   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�        max       @FQ��     p   4   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @ve\(�     p  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P@           l  1   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�8        max       @�n           �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �T��   max       >�$�      �  2X   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�   max       B/_      �  30   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B/F^      �  4   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >`�9   max       C�4      �  4�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��Y   max       C�      �  5�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          S      �  7h   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5      �  8@   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�Y   max       Pn��      �  9   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?����)_   max       ?�ح��U�      �  9�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �ě�   max       >�+      �  :�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @Fz�G�     p  ;�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @ve\(�     p  D   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P@           l  L�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�8        max       @��           �  L�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         C�   max         C�      �  M�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�=�K]�   max       ?��
=p��     �  N�      	      c   �               �            &      "      +   -   G            $   	      "   �      M                  	         R   	   �                           �      >   B   N�=>N��O�vP�qP��~M�YN*:qO
�,N=�DP���N��N��OᚰO���N"�'O̹*M�AO�!�P@�P
K�M�^�Ok�O�`ON�OatO���P&:BN���P;l�ONI�N�.�N�ngNEJGN vNҫ5O�#O��P�N��P��0M�Y:O!�yOK�*N}�bOG��O�nNX��N��GO�ŅN6i{O�ݝOA-�Nj �������1�e`B�ě����
�D���D���o:�o;o;�o;��
;ě�;ě�;�`B<49X<D��<T��<e`B<e`B<u<�C�<�t�<���<��
<��
<��
<�1<�1<�1<�/<�h<�h<�h=+=�w=#�
=#�
=49X=L��=P�`=aG�=q��=�C�=�hs=���=��w=��
=��=��>�>�>\)>�+��������������������ggkltu������tggggggg}~�����������������}��)BJBJRPTSF)���SW����6B)�����zgaS!##$0300#!!!!!!!!!!##024200+#JGN[gtx�|~����tg[PNJeght����ytqheeeeeeee)B[����������g����������������������������������������^_emz�����������zma^�������������������������������������/;GLKLHC:/""/��������������������!)<HUanz|xnl`H</*##!)[kogf_WON5����
/<BE@</#
������������������������������
 #&#
�����! #(1<DHUcb[QH</*#������
<UH</$"����EFHILU[acebaUHEEEEEEzywx}�����������zzzz�����
!#
��������������������������)*66:;:6)(56BEO[��������h[QFC5��������������������f_dgt�����������ztgf��������������������fghqt����{thffffffff������������������������� ��������������������������"56>BC6*12;@Uam�������zmTH;1!"/0;AHTahiaZTH;/"!!|������ ���������||	 ��������������������(%')5<BO[^cdb[SOB6)(��������������������65;<>DHJQUahnrhaUH<6`YWX]amz������zzma``<8<<HIUZ`UUI><<<<<<<#/2685/(#########�����

��������
 	
�������#%"������������

������������������������čĚĦĳĻĿ����ĿĶĳħĦĚĒčĈăččÇÇÓàììöìàÓÇ�|ÀÇÇÇÇÇÇÇ�N�[�^�`�[�S�N�D�B�5�)��$�)�+�3�5�B�C�NŔŠŽ������ŹŔ�{�U�0�
�����������0�bŔ������������������N���(�Z�~�����˻F�S�_�`�_�X�S�F�D�C�F�F�F�F�F�F�F�F�F�F�x�������������x�l�j�l�m�x�x�x�x�x�x�x�x���������ʾ;ʾ����������������������������������û������������������������������B�[�tčĘē�~�h�O�?�;�)�#�
������B�׾����	�����	�����׾ʾȾʾ˾Ѿ׺'�3�9�:�5�3�'������'�'�'�'�'�'�'�'�(�5�N�[�f�j�n�j�Z�N�A�5�(�#������(�ּ��������������ּм̼ɼʼּ̼���&���������������������m�a�H�;�/�"����!�H�T�a�m�v�z�����/�<�E�?�<�4�/�+�-�.�/�/�/�/�/�/�/�/�/�/��������������������������������ѿ����)�;�=�3�(�����ѿĿ������������������������y�`�G�8�+�%�&�!�+�T�y�������������������������������������������������&�&��������������������"�/�;�G�T�a�i�m�q�y�a�H�;�"��	����"�<�H�U�n�w�{ÇÊ�~�y�g�I�G�<�/����#�<�����)�3�)�������������������������������������������{�z�}���������4�=�E�M�_�c�\�M�A��н������ýнݽ��4�����0�B�F�4���ֺɺ��������������ɺ�ʾ׾�������׾ʾ��������ɾʾʾʾʾ����������������������Z�J�>�<�B�M�f�s�����'�5�3�(��������ݿοѿ׿�����m�y�������������������������y�s�m�b�j�m���(�4�<�4�4�(�'����
��������-�:�;�F�L�N�F�:�4�-�#�*�-�-�-�-�-�-�-�-���ùϹܹܹܹϹƹù�������������������������������������������������������������Óàäìùú��������ùìàÓÎÊÉÑÓÓ�G�S�_�`�j�e�`�S�R�G�:�.�$�&�-�.�8�:�D�G�tāēĦĿ������ķĦĚč�}�j�d�e�p�e�g�t�#�$�,�0�9�<�@�C�>�<�0�+�#� ���� �#�#�4�D�S�S�G�0���ܻ������x�p�s�������4�m�y���������y�v�m�f�m�m�m�m�m�m�m�m�m�m�g�s���������������������������t�g�d�b�g�~�����������ɺкɺ��������������}�u�{�~ù����������ùïìèìôùùùùùùùù��(�A�E�M�Z�`�Z�M�J�=�4�(�������ŔŠŭŹ����������žŹŭšŠŔœŐœŔŔ�ֺ��������������ٺֺպֺֺֺֺֺ�ǭǩǡǔǈ�{�u�{ǀǈǔǡǭǭǭǭǭǭǭǭD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��I�<�0�&�#�"�#�0�<�I�J�I�I�I�I�I�I�I�I�I¦²¿����������������¿²¦¦EuE�E�E�E�E�E�E�E�E�E�E�E�EuEtEpEqEpEkEuE7ECEPE\E^EbE\EPECE>E7E2E7E7E7E7E7E7E7E7 8 N ^ Z L 7 6 N @ \ S I : S + 8 � S G , / 3 D G ;   f + F G J n 5 K V J + Z : o T M < J 7 8  [ K ' 8 3 1 O  
  �  `  �  	   �  F  p  Z  �  0  �     z  =  �  M  E  4  �    ,  C  &  �    �  �  �  Z  �  ;  �  g  @    &  n  �  [  �    ~  �  �  �  T  �  �    ^  �  �  ��T���T���o=�v�>��%   ;D��;o<t�>8Q�<49X<�o=o=0 �<��
=8Q�<�o=e`B=m�h=�1<�1<�<��=aG�<�h<�=]/>1&�<�h=Ƨ�='�=t�=\)=\)='�=D��=�C�=q��>%=q��>S��=m�h=��=�^5=��-=���=�v�=�j=�v�>�$�>\)>C��>Q�>+B�CB	��BXxB`aB![B%s�B%��B	d�BCBw�B!f�B!8RB lB�B!��A�B K@B��BNdB��B �5B��B��B�~B�<B8�B$/2B�BBwB�BA\B
$QB"�BF�B˅B.>�B!��B/_A��OA�/�BlB^�B�4BHB�B�xA��,B&�AB�B�_B��B�2B4�BFyB�9B	��BGB@B�"B%a�B%��B	�B@	BN_B!D�B!<�A�}�B0tB!��A��B �YB�Bb�B��B �;BW�BA�B��B�#B?�B$@�B�UB@"B2�BD�B	�[B8�B�B�?B.��B"?�B/F^A��CA��4B@	BD�B��B:EB±B�WA��,B&�ZB��B�tB�VBB�BK[B@�A�^�A��jA��A��:A��v@��"@�N�AL3�@���A��AV��?���A� A߫@æaA��A��AҟdA� PAj_7@�!YA�wA���A��A��@A /�A2�@L��AR��ADC�A�v�Ao(A5]/@{m;>`�9At8�A��2A�UAߒGA�@��Am,�A���@�hA���A7�iA���@F�3B?]C��|A��A��C�4C���A���A��nA�~�A�~LA��@��v@�AML�@�!}AْKAUf�?�?�A���A��@�tA��kA�~�AҀTA��Aiw�@���A�s�A�`�Aƀ"A� A<�A5�,@L�AS0AD�A��8AmH�A5��@���>��YAt��À�A 
A߰�A�}@��An��A�>q@AA�sKA6�A�r�@K��B@�C���A�nA���C�C���      	      d   �            	   �            &      #      ,   -   G            %   
      #   �   	   M               	   
         S   
   �                           �      >   B               A   S               ;         #         !         -   '            )         1   +      /                           '      @                                                   /   5                                          +                        +                                          -                                       N��GN��O�vPD��Pn��M�YN*:qO
�,N=�DO���N��N4�CO�fOr�pN"�'O��M�ANX�tP7�OmfM�^�N��O�`N�o�Np �OatO��O�LN[�@OHE�ONI�N�.�N�ngNEJGN vNҫ5O�#O��O�_�N� aP)�!M�Y:O!�yOK�*N}�bOG��O�nNX��N��GO$�(N6i{O�ݝOA-�Nj �  �  �  U  A  
-  �  �  �  p  �  ~  �  =  l  �  ;  �  �  P  	^           �  �    �  A    �  Z  �  �  �  �  
  �  9  �  �      ?  -  �  �  6    �  �    i  �ě���1�e`B<�/=H�9�D���D���o:�o=��`;�o;�`B<49X<#�
;�`B<���<D��=\)<��
=,1<u<�t�<�t�=�P<�1<��
<�1=�E�<�j=u<�/<�h<�h<�h=+=�w=#�
=,1=�t�=T��=�E�=aG�=q��=�C�=�hs=���=��w=��
=��>n�>�>�>\)>�+��������������������ggkltu������tggggggg}~�����������������}����)*2=@DB5)����������� �������!##$0300#!!!!!!!!!!##024200+#JGN[gtx�|~����tg[PNJeght����ytqheeeeeeee/-.25BN[gny}|tg[NB5/����������������������������������������ebbimz�����������zme�������������������������������������	
"/;@EDBCA<2/"	��������������������./4<HLSHE<0/........)N`hjcaWNB5)����
/9=>:4/#
��������������������������
!
�������! #(1<DHUcb[QH</*#����
##$#
���HHJMUZabda_UIHHHHHHHzywx}�����������zzzz����
 #"
� ������������������������)36886) fhmty�����������ztkf��������������������f_dgt�����������ztgf��������������������fghqt����{thffffffff������������������������� �������������������������*36=@@6*`XTU\akmz��������zm`-/4;FHTXaWTH;/------��������������������	 ��������������������(%')5<BO[^cdb[SOB6)(��������������������65;<>DHJQUahnrhaUH<6`YWX]amz������zzma``<8<<HIUZ`UUI><<<<<<<#/2685/(#########�������

	������
 	
�������#%"������������

������������������������čĚĦĳĳĿ��ĿĿĳĦĚĕčċČččččÇÇÓàììöìàÓÇ�|ÀÇÇÇÇÇÇÇ�N�[�^�`�[�S�N�D�B�5�)��$�)�+�3�5�B�C�N�0�I�{ŔŚťūŠŔ�{�I�#�
������������0���������������������������p�N�@�A�N�s���F�S�_�`�_�X�S�F�D�C�F�F�F�F�F�F�F�F�F�F�x�������������x�l�j�l�m�x�x�x�x�x�x�x�x���������ʾ;ʾ����������������������������������û������������������������������B�O�[�h�m�w�y�v�n�h�[�O�B�:�4�2�1�3�9�B�׾����	�����	�����׾ʾȾʾ˾Ѿ׺'�3�7�8�3�3�'����'�'�'�'�'�'�'�'�'�'��(�5�N�T�_�_�a�e�b�Z�N�5�(�&������ּ��������������ټּӼμͼͼּ���&�����������������T�a�p�x�|�u�m�a�T�H�;�/�"����"�;�H�T�/�<�E�?�<�4�/�+�-�.�/�/�/�/�/�/�/�/�/�/��������������������������������������4�8�0�(������ѿĿ����������Ŀݿ���m�y�����������y�m�`�T�G�@�<�A�G�N�T�`�m���������������������������������������������$�%��������������������"�/�;�G�T�a�i�m�q�y�a�H�;�"��	����"�U�a�n�o�{�}�z�o�n�a�_�U�I�H�<�5�<�H�K�U����)�1�)������������������������������������������{�z�}���������4�A�M�^�b�[�M�A�(��н������Ľнݽ��4�������������ֺɺƺ��úκֺ����׾����޾׾ʾ����ʾо׾׾׾׾׾׾׾׾s�������������������s�f�`�Y�W�Z�]�f�s���'�5�3�(��������ݿοѿ׿�����m�y�������������������������y�s�m�b�j�m���(�4�<�4�4�(�'����
��������-�:�;�F�L�N�F�:�4�-�#�*�-�-�-�-�-�-�-�-���ùϹܹܹܹϹƹù�������������������������������������������������������������Óàäìùú��������ùìàÓÎÊÉÑÓÓ�G�S�Z�`�c�b�`�Y�S�P�G�:�.�&�(�.�.�:�F�GāčĚĦĳĸĿ����ĿļĲĦĚčā�y�u�wā�#�%�0�6�<�=�>�<�0�#�"���#�#�#�#�#�#�#��'�4�=�E�G�<�'����û��������л����m�y���������y�v�m�f�m�m�m�m�m�m�m�m�m�m�g�s���������������������������t�g�d�b�g�~�����������ɺкɺ��������������}�u�{�~ù����������ùïìèìôùùùùùùùù��(�A�E�M�Z�`�Z�M�J�=�4�(�������ŔŠŭŹ����������žŹŭšŠŔœŐœŔŔ�ֺ��������������ٺֺպֺֺֺֺֺ�ǭǩǡǔǈ�{�u�{ǀǈǔǡǭǭǭǭǭǭǭǭD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��I�<�0�&�#�"�#�0�<�I�J�I�I�I�I�I�I�I�I�I¦²¿����������������¿²¦¦EuE�E�E�E�E�E�E�E�E�E�E�E�EuEtEpEqEpEkEuE7ECEPE\E^EbE\EPECE>E7E2E7E7E7E7E7E7E7E7 > N ^ e / 7 6 N @ . S I = O + > � 6 H ) / 2 D ; D   e ' 4  J n 5 K V J + Z   e 6 M < J 7 8  [ K  8 3 1 O  �  �  `       �  F  p  Z    0  d  c  
  =  8  M  g  �  �      C    k    �  �  i  �  �  ;  �  g  @    &  C    �  !    ~  �  �  �  T  �  �  V  ^  �  �  �  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  C�  �  �  �  �  �  �  �  �  �  f  H  (    �  �  �  �  O    �  �  �  �  p  [  F  3      �  �  �  �  �  o  J  !    �  �  U  S  Q  O  K  F  ?  3  %    �  �  �  �  �  Z  )  �  �  a  �  )  z  �    ,  @  :  0    �  �  j    �  O  �  �  �  �  C  !  �  	M  	�  
  
'  
*  
  	�  	�  	�  	(  �  A  z  n  W  �   �  �  �  �  �  �  |  x  r  k  c  \  U  N  B  +    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    l  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  d  U  E  5  &  p  z  �  �  �  �    ~  }  |  {  y  w  r  m  h  b  =  �  �  
�  �  �  2  �    a  �  �    �  �  �    0  (  �  
  �  ?  ~  y  t  n  e  \  P  C  7  '      �  �  �  u  D  1  )  "  |  �  �  �  �  �  �  �  �  �  �  s  ]  @  #    �  �  �  �  �    *  ;  <  8  -    �  �  �  |  �  }  I    �  N  �  A  N  d  l  j  e  ]  P  9    �  �  c  	  �  H  �  z  �  L  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  T  7    �  �    *  8  9  .      �  �  �  J    �  H  �    M  \  �  �  �  �  �  �  \    �  �  f  Q  <  (     �   �   �   �   �  u  �    -  I  r  �  �  �  �  �  �  ^    �  �  >  q  �  �  �  -  I  O  7    �  �  �  �  �  �  �  r  8  �  �    K  8  �  A  �  ^  �  	  	@  	Y  	\  	E  	  �  ]  �  e  �    �  �  �        �  �  �  �  �  �  l  J     �  �  �  n  ?    �  �  {  ~  |  x  o  e  X  I  9  '       �  �  �  ~  V  '  �  �          �  �  �  �  �  �  �  x  x  �  t  H    �  �  z    O  T  T  P  e  �  �  �        �  �  a    �  ;  �  {  �  �  �  �  �  �  �  �  �  �  o  N  *    �  �  �  s  L  $  �  �  �  �  �  �  �  w  k  ^  J  /    �  �  �  C     �   �    
    �  �  �  �  �  �  �  a  2  �  �  \    �  p    �  
�  W  B  �    �  @  �  �  �  �  h    �  �  �  o  	�  �    7  :  >  @  A  @  <  8  +    
  �  �  �  �  h  8     �   �  �  .  �     W  �  �  �          �  �  p  
  k  �  �  #  �  �  �  �    c  E  %    �  �  �  �  s  ^  J  &  �  J   �  Z  U  P  I  =  1  -  /  0  /  -  *  "          
      �  ~  s  h  c  `  ]  X  S  N  E  8  *    �  �  �  �  [  0  �  �        $  1  D  Z  p  �  �  2  m  �  �  �  �      �  �  �  �  �  �  �  j  J  +  	  �  �  �  `  )  �  �    G  �  �  �  r  c  U  G  :  ,           �  �  �  �  Z  (   �  
  �  �  �  d  .  �  �  �  �  ^  ;  #    �  b    �  -  �  �  �  �  �  �  m  W  ?  %  	  �  �  �  �  ^    �  w  �  �  	  	�  
d  
�  
�    5  5     
�  
�  
�  
3  	�  	  E  `  M  T  �  �  �  �  �  �  �  �  �  �  z  g  Q  :  &      �  �    �    T  �  H  u  �  s  :  �  l  �  C  l  x  
b  	  �  a  !  �    	    �  �  �  �  �  �  �  �         �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  k  Q  7       �  �  ?    �  �  �  �  z  R  ,    �  �  K  �  �  d  @  "  <    -  )  %  !        �  �  �  �  �  �  v  S  0    �  �  �  �  �  �  �  �  �  �  v  ^  B  !  �  �  �  U    �  �  .  �  �  n  R  9  "    �  �  �  n  =  
  �  �  d  #  �  n  �  k  6    �  �  �  �  j  A    �  �  �  �  l  A    �  �  |  G    �  �  �  �  �  q  V  :    �  �  �  �  `  I  =  @  �  �  �    A  r  �  �  �  �  r  .  �      �  �    :  �  �  �  �  �  �  K    �  �  �  |  Q  %  �  �  �  c  ,  �  �  |  :    �  �  �  �  �  \  '  
�  
�  
+  	�  	<  �  �  +  1  5  ]  �  i  \  D  ,    �  �  )  �  :  �    W  �  
�  	�  �  c    �  �  �  �  S  *  �  �  }  :  �  �  E  �  C  �  �  @  �  �  M