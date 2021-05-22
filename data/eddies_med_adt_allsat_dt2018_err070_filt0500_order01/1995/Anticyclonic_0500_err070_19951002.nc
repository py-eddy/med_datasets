CDF       
      obs    :   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�(�\)      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N 5�   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �t�   max       =�l�      �  |   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @F�����     	   d   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���G�|    max       @vt�����     	  )t   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P            t  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�v           �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �D��   max       >�l�      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�ϔ   max       B-��      �  4�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�m�   max       B,�      �  5�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�+   max       C�x�      �  6�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�xR   max       C�zR      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         <      �  8h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      �  9P   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5      �  :8   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N 5�   max       Ps3n      �  ;    speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��t�j~�   max       ?�����      �  <   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �t�   max       >D��      �  <�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?W
=p��   max       @F�����     	  =�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @vt�����     	  F�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @O�           t  O�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��          �  Pl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         AF   max         AF      �  QT   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�hr� Ĝ   max       ?��1���.        R<         
   "      
   
   1   c  ;   (            Q   
      0      !   
         (                     9            
   2               
   V         O      
      (      �   Z      
      .      N���N죕N 5�O�E�N�N�E�N�L O���PXQ�P��P�Pw�O�2�O�eP�e�N��yOL�CO��oN��CO��hN]��N���O��O��tNN8<O�7N��N&��O��O�Ps�N�ֻO�O�N���OB;O���O��O
�N/M'N�f:Oޥ1N5SN��P#��O��N��Ob�@O�s\O�/�O���O��N���NB�NJ��O3�kN,S�NHpr�t��ě���o��o�o��o<t�<#�
<49X<D��<e`B<e`B<�o<�C�<�C�<�t�<�t�<�1<�9X<�9X<���<���<�/<�/<�`B<�h<�<�<�<��=#�
=#�
='�='�='�=49X=8Q�=D��=H�9=H�9=T��=Y�=]/=]/=aG�=ix�=m�h=m�h=q��=u=u=�7L=��P=���=���=��=��=�l�lgfgmnz����znllllll�������������������� ����
#06.0-
��������������������������d^^gpt��������tgdddd/45BFN[\bghge[NB95// #/HUaknpoplcU</'$ ������	)07;95)����	5Nt��������gO5xy{����������������x�����)5NSMIEIB5#����������
#'��������������	��������������!1782)�������#&00100#!#)0<ISUXUTKI<0-*'#!����������������������� ��������������������������IORZ[hppkh[OIIIIIIII~��������������~~~~D=>CHUagnz�{tona^UHD��������������������~���������~~~~~~~~~~BNN[giigf[VNB@5215?B�(	����������4)05:BIIB54444444444����������������������������������������ejuv�����������te#/<HLQHC</*#/;;<HTamrvsmaTH;////
#(/;?>A<#����������������������������������������DGIB<
������
#/<D���#+1420'*)������

�������		
 
										yz������������zzyyyynbqz��������������}n��������������������HHIUacaaWUTHHHHHHHHHGFOW[htu��������tbOG����!*.1)��&)-0255)'
)5;?BFEDB:5)!������*5:0*����� %)5BNTY[XRNB)!���������

��������������
������)+6BO[][UOKB6*)<30+),0<@AD<<<<<<<<<���������������������������

�����#).#nvz�����zsnnnnnnnnnnE�E�E�E�E�E�F
E�E�E�E�E�E�E�E�E�E�E�E�Eͺ�����������������������������������������������ùùòù�����������������������޻��������������x�_�S�F�:�-�.�:�F�S�l�������������������������������������������������������������������������������������n�{�}ńŇŊŉŇ�{�p�n�i�b�[�W�[�b�f�n�n�����#� ���"��������������������
�I�U�d�x��n�X�I�0�������������������
�6�O�hčĢĮģčā�O�6������������6�Z�s�����������s�g�T�N�A�<�(�#� �%�5�A�Z���(�5�N�Z�n�����k�Z�(����ѿ������Ŀݿ����� � �*����������ݿٿտֿοݿ��)�6�<�B�O�^�e�c�[�B�6�+�$�)�6�:�6�)�'�)���/�_�a�\�T�;�"����������������������ʼּ�������ּʼɼüǼʼʼʼʼʼʼ������������������z�r�f�d�`�^�f�r�����àìù�����������������ùìâÛÛÙà�m�z�������������z�m�a�`�a�c�m�m�m�m�m�m��� �������ݽĽ�������������Ľ߾�H�M�U�a�c�i�a�U�H�F�@�B�H�H�H�H�H�H�H�H�<�H�J�Q�S�H�@�<�1�/�/�+�(�)�/�6�<�<�<�<����(�3�4�3�5�9�;�5�(������ ����A�N�Z�a�v�n�l�u�n�a�Z�5�(�#�� �'�2�5�A�������ĿĿ����������������������������������߽ݽнǽĽ������������Ľнӽݽ�Z�^�_�[�Z�M�F�J�M�R�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z��(�4�7�4�0�(��������������4�A�M�Z�f�t�y�x�m�f�Z�M�A�4�+�$� �"�*�4�G�T�V�`�e�`�`�`�c�_�T�G�?�=�;�8�:�;�?�G��������"�B�Q�N�;�"��	������������������"�%�+�+�%�"���	���	�������čĖĚĦĮįİĬĦĚčĆąćĉČčččč�Z�s������������������������s�f�[�S�U�Z������������ܻػӻܻ���� ��àÓÏÇÂ�}�~ÇÓàìùý������ûùìà�g�[�N�B�;�)�1�N�[�g�t�~�u�q�t�g�;�G�T�_�`�T�;�.�"��	�����������.�;�.�7�;�B�C�;�4�.�"��	�����	��"�#�.�.�T�`�m�n�r�m�`�T�I�P�T�T�T�T�T�T�T�T�T�T�
���������
�������������
�
�
�
���ɺ�����������ֺɺ��������������
�����
�����������������������������(�3�+�(����������������'�M�r��������r�Y�M�@�'��������F�S�[�_�l�n�l�`�S�M�F�:�-�!��!�%�.�:�F������������¿²¦¦²¿�������B�N�[�p�t�g�`�N�B�5�0�)�'�'�)�+�5�B�������������������y�l�`�]�[�\�[�\�[�{���������$�-�8�:�0�$��	����������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DwDvDD�D����ùϹܹ���	��������ܹù����������'�3�5�?�3�*�3�6�6�3�1�'�������%�'�������ɺֺ����ֺɺ���������������������������������������������������������EuE�E�E�E�E�E�E�E�E�E�EuEiEgEdEeEeEiEiEu��*�6�;�6�3�*��������������ּ����ּʼż¼ʼּּּּּּּּּ� Z ; r 2 [ < 0 + : 1 K O I J 6 # ' & M P ^ V R ? I 0 u R ) [ C Z ^ M b  5 M 0 ? d H f - A \ s 5 i = % A � 5 l . N "  �     S  �  P  "  '  X  �  0  �  >  �  2  �  �  �  �  �  �  �  �  i  �  ~  -  w  )  a  +  (  �  �  J    �  m  H  +  ?  �    )  7  �  �  T  �  x  8  R  C  G  ^  K  �  <  ^�D��<�C�;ě�<�h;�o<t�<���=q��=�G�>�l�=Y�=��=,1=49X=ȴ9<�/=C�=�C�<�/=aG�=\)=�P=C�=�+<��=C�=+=\)=P�`='�=Ƨ�=<j=T��=�+=L��=�v�=��P=�C�=�o=e`B=y�#>O�=m�h=q��>	7L=��=�C�=��-=ȴ9=�E�>Kƨ>�w=�{=�j=�`B>�P=�S�>+B�B!�2B�+B$S?B�SB
�B-�B	�B�8BmB�sB�5BS�B�BҁB%�B&CgB�B�B!y�BI6B��BtB�BS�B<�B��Bz�Bo4BX�B �B\�A�ϔB|nB�B"^By�BebB�iB%�B8�B�B�B�aB'tBV�B1�BݢB-��B�gB�^B%�B�B&B�BD�B��Bd0B�+B!��B?�B$B�B�B
AB��BŉB��B@�B@{B9LB�jB��B>B%8~B&?vB�@B�B!��BB�B�=B�CB:&BHtB{�B�8BC?BB8�B@mB��A�m�B�]BpB"A,B��B?�B�B@HBwGB7�B��B��B?�B�qB@B� B,�BC�B�FB��B�NB%��B@"B��B�sBCC�x�@��A��p@�ӈA��dA���A��Aң�A�o�A�A�A�\�A�ٴA�	�A�*YA�y;A�@�9LA�\�A��xA*[A�OPA�0�A��rA�6Au�mA(ΨA>@~A6�A=zAeԢA���A��>A�r8AF[@�E�A˩�A�X^Aa%�A^�NAh��A���@E��A�nRA5��@���@�N?A��A��,A�B��C��V>�+?��"@8�t@�cC��A�R�A ��C�zR@�A�w�@���AБ�A�~�A��>AҀ�A�Z�A��A�RA�#aA�iuA�k�A��A��@�<A��A�GA+�A�}(A�z�A�8(A�P|Au��A(k�A>��A6��A<��Ae,A�~SA�v-Aߎ�AG c@�/MA�7�A�cRAaFA_�AiCA��7@Kb�A���A5��@� @�ʋA���A��A	�B�nC���>�xR?�5�@;��@��6C���A�vA �            "            2   d  <   (            R   
      0      "            (                     :            
   2               
   V         P            (      �   Z      
      .                  #               /   ?   )   7   #      ?               +            !                     5                     !            #         +            %         #                                                   !   5         3               #                                                                           )            %                           N���N�S�N 5�O�ON�N�E�N�L N��O��O���O���P^� O��N�JPs3nN��yOL�CON�PN��CO��{N9k|N|-�O��O8��NN8<O�7N��N&��O��O�O���N�ֻN�&O�N���O�DO�T�Oc�1O
�N/M'N�f:Oh��N5SN��O��CO��N��OSuXO�s\OBMOf�O���NY�NB�NJ��O3�kN,S�NHpr  �  �  �  H    �  x  �  
,  �  ?  1  U  .  l  (  v  Q  �  �  ^  �  �  �  �  �  0  w  �  �  �  %  f  �  }  �  C  �  �  *  2  �  9    �  
  �  �  S  i  �        :  
  �  ��t��D����o%   �o��o<t�<��=P�`>D��<�9X<�o<�1<�<�<�t�<�t�=�w<�9X<�h<���<�/<�/='�<�`B<�h<�<�<�<��=�%=#�
=,1='�='�=P�`=<j=L��=H�9=H�9=T��=��T=]/=]/=�7L=ix�=m�h=q��=q��=�7L=�x�=��T=��-=���=���=��=��=�l�lgfgmnz����znllllll�������������������� ��������#-23*,*#	����������������������d^^gpt��������tgdddd/45BFN[\bghge[NB95//-./5<HIUW]XUH<0/----�����(./.)����.,.5BN[gluxwtkg[NB5.�~���������������������� )5NPFBGA5)����������
��������������������������������'12-)�����#&00100#!#)0<ISUXUTKI<0-*'#!����������������������� ��������������������������LOS[[hnohh[OLLLLLLLL��������������������D=>CHUagnz�{tona^UHD��������������������~���������~~~~~~~~~~BNN[giigf[VNB@5215?B�(	����������4)05:BIIB54444444444������������������������������������������������������������#/<HLQHC</*#@>>HTamqurmaTH@@@@@@
#(/;?>A<#���������������������������������������������
#/<DGHA<
�����!$)/2/)�������

�������		
 
										yz������������zzyyyy����������������������������������������HHIUacaaWUTHHHHHHHHHNJJKU]ht}�������thVN����!*.1)��&)-0255)'
)5:>BFECB75)#������*5:0*�����$#$)35BFNTVSNLGB51)$��������	

	����������������������� ")*6BOROFB61)      <30+),0<@AD<<<<<<<<<���������������������������

�����#).#nvz�����zsnnnnnnnnnnE�E�E�E�E�E�F
E�E�E�E�E�E�E�E�E�E�E�E�Eͺ�����������������������������������������������ùùòù�����������������������޻S�l�������������������x�_�S�F�@�;�:�F�S���������������������������������������������������������������������������������n�{�}ńŇŊŉŇ�{�p�n�i�b�[�W�[�b�f�n�n����������������������������������#�0�<�I�K�I�@�0��
���������������
��O�[�h�t�}āĀ�y�t�h�[�O�A�6�3�1�2�6�A�O�A�N�Z�s��������n�g�Z�N�A�5�(�&�'�+�2�A�����5�N�Z�k�{�|�g�N����ѿ������Ŀݿ������#�����	�����ݿڿܿܿ޿���B�O�U�Z�R�O�B�A�;�B�B�B�B�B�B�B�B�B�B�B�	��/�P�W�X�T�I�;�"�������������������	�ʼּ�������ּʼɼüǼʼʼʼʼʼʼ������������������z�r�f�d�`�^�f�r�����ù������������������������ùòìèéîù�m�z�������������z�m�a�`�a�c�m�m�m�m�m�m���������нĽ��������������Ľн��H�K�U�a�b�g�a�U�H�H�B�E�H�H�H�H�H�H�H�H�/�<�H�M�L�H�=�<�/�,�*�+�/�/�/�/�/�/�/�/����(�3�4�3�5�9�;�5�(������ ����A�N�Z�]�`�a�f�a�Z�N�A�5�-�(�&�(�(�.�5�A�������ĿĿ����������������������������������߽ݽнǽĽ������������Ľнӽݽ�Z�^�_�[�Z�M�F�J�M�R�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z��(�4�7�4�0�(��������������4�A�M�Z�f�t�y�x�m�f�Z�M�A�4�+�$� �"�*�4�G�T�V�`�e�`�`�`�c�_�T�G�?�=�;�8�:�;�?�G���	��*�/�6�8�5�/�"��	������������������"�%�+�+�%�"���	���	�������čĚĦĬĭįīĦĚčććĈċčččččč�Z�s������������������������s�f�[�S�U�Z������������ܻػӻܻ���� ��Óàìøù����ù÷ìàÓÇÆÁÃÇÏÓÓ�[�g�t�}�t�q�s�g�[�N�B�<�*�2�J�[�;�G�T�\�^�\�T�G�:�.�"������	��"�.�;�.�7�;�B�C�;�4�.�"��	�����	��"�#�.�.�T�`�m�n�r�m�`�T�I�P�T�T�T�T�T�T�T�T�T�T�
���������
�������������
�
�
�
�ɺֺ�������������ֺƺ������ź����
�����
�����������������������������(�3�+�(�����������������'�4�M�r����r�Y�M�@�'���������F�S�[�_�l�n�l�`�S�M�F�:�-�!��!�%�.�:�F������������¿²¦¦²¿�������N�[�m�}�~�t�g�^�N�B�5�2�)�(�'�)�,�5�B�N�������������������y�l�`�]�[�\�[�\�[�{�������$�0�2�3�0�%��������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D����ùϹܹ����
��
����ܹù������������'�3�9�3�)�0�2�'�!������'�'�'�'�'�'�������ɺֺ����ֺɺ���������������������������������������������������������EuE�E�E�E�E�E�E�E�E�E�EuEiEgEdEeEeEiEiEu��*�6�;�6�3�*��������������ּ����ּʼż¼ʼּּּּּּּּּ� Z / r . [ < 0 ! -  8 S 9 0 2 # ' * M L ^ R R . I 0 u R ) [ ! Z G M b  5 ? 0 ? d 8 f - ; \ s 4 i 4  < ~ 5 l . N "  �  �  S  k  P  "  '  �  �    �    &  =  �  �  �  �  �    u  �  i  �  ~  -  w  )  a  +  �  �  #  J    A  Y  �  +  ?  �  �  )  7  A  �  T  �  x  �  J  �  �  ^  K  �  <  ^  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  AF  �  �  �  �  �  �  �  �  �  |  w  v  t  f  ?    �  �  �  l  �  �  �  �  �  �  �  �  �  �  h  G     �  �  Q  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  ;  �  `  C  %    �  �    /  E  @  2  !    �  �  �  �  �  }  ]  ,    �    ~   �                      	    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  g  S  >  %    �  �  �  �  �  S  �  x  e  S  A  /         �  �  �  �  �  �  u  Y  A  )  �     �  U  �  �    F  v  �  �  �  �  V    �  z  	  s  �  +  �  �  �  	  	�  	�  
  
  
)  
+  
  	�  	�  	I  �  X  �  *  =  �  U  �    -    �  (  w  �  q  
  d  �  N  �  �    �  �  B  :  �    .  ;  ?  6    �  �  �  �  �  �  U  !  �  �  Z  �  ~  $  /  0  -  "    �  �  �  �  �  �  {  ]  F  $  �  �  J    6  5  A  S  T  K  :  "    �  �  �  �  �  �  H  �  �  1  �  �  �  ~  �  �  �  �  	    ,  *      �  �  �  X    �  �  �  [  k  l  j  ]  9    �  �  v  Q  +  �  �    S  �  �  	  (  &  %        �  �  �  �  �  }  e  O  ?  1  .  +  !    v  e  Q  :  (    �  �  �  �  �  z  o  h  L  2    �  �  �  M  �  �    $  ;  J  Q  I  *  �  �  k    �  �  P  �  �  I  �  }  s  j  _  P  A  3  )  (  (  '  8  U  s  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  w  l  X  ?    �  �  $  A  M  V  _  k  v  |  �  �    r  ^  A  &    �  �  �  �  s  P  �  �  �  �  �  �  �  �  �  �  �  o  W  ;    �  �  �  j  :  �  �  �  �  �  �  u  h  Z  H  5  "    �  �  �  �  �  �  p    @  j  �  �  �  �  �  �  �  �  k  >     �  b  �  R  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  S  5     �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  m  j  j  k  k  0  )  "          
      &  /  8  =  8  2  -  (  #    w  l  a  V  L  F  @  :  0  !      �  �  �  �  �  �  ~  j  �  �  �  �  �  �  �  �  �  q  \  D  %  �  �  �  o  @  '    �  �  �  �  �  �  �  �  v  l  j  p  n  f  X  G    �  �  �  H  ;  /  m  �  �  �  �  �  �  �  �  n  >  �  �  3  �      %      �  �  �  �  �  �  �  z  c  L  3    �  �  �  �  a  %  K  e  `  X  N  ?  0  "      �  �  �  �  �  ]  5    �  �  �  �  �  r  j  c  W  F  *    �  �  �  H  �  6  �   �   U  }  n  _  K  8    �  �  �  �  a  G  >  ,    �  �  �  O    �  �  �  �  �  �  �  q  J    �  �  _  �  7  �  �  	  B  �  >  B  ?  ?  ?  9  ,    �  �  �  y  <  �  �  7  �  ]  �  �  �  �  �  �  �  �  �  �  �  �  l  E    �  �  z  ;  �  S   �  �  x  j  Z  H  5      �  �  �  �  [  5    �  �  �  q  >  *        �  �  �  �  �  �  �  �  u  `  M  <  +    �  �  2  *  "      �  �  �  �  �  d  ;    �  �  �  �  b  ;    
[    V  �  �  �  �  �  �  {  D  
�  
�  
9  	�  �      �  �  9  9  9  8  8  3  "       �  �  �  �  �  �  p  Y  A  *          
    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    W  z  �  �  i  ;  8  O  5    �  �  `    �  �  R  �  @  
    �  �  �  �  �  s  T  5    �  �  }  >  �  f  �  ]   �  �  �  �  �  �  �  |  q  e  U  A  (  �  �  �  H  �  �  U   �  �  �  �  �  �  �  e  <    �  �  w  9  �  �  v  e  <    �  S      �  �  �  �  �           �  �  �  x    �  -  o  >  H  R  c  i  g  [  F  /    �  �  �  e  '  �  �  >  �    �  q  �  5  r  �  �  �  �  �  p  �  [  {  z  F  �  T  �  	�  �  �          �  �  �  <  
�  
|  

  	�  �    %  
  �  �  �  �  �             �  �  �  w  &  �  �  {  |  �  �  ,      �  �  �  �  �  �  �  n  [  F  0      �  �  f  "  �  :  8  ,    �  �  �  y  Q  +      �  �  |  A    �  �  �  
  
  

  	�  	�  	�  	�  	n  	6  �  �  L  �  n  �  -  y  �  G  �  �  �  �  �  �  �  �  �  �  {  t  l  _  M  <  *  �  �  �  L  �  n  ]  I  /    �  �  �  ;  �  �  T    �  \    �  L  �