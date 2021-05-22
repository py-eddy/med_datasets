CDF       
      obs    3   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?ѩ��l�D      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�ۄ   max       P���      �  x   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �D��   max       =�9X      �  D   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�\(��   max       @F5\(�     �      effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v�          �  (   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @Q`           h  0    effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���          �  0h   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �o   max       >�z�      �  14   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B0Y�      �  2    latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�   max       B0�+      �  2�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       @̶   max       C���      �  3�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =��D   max       C��      �  4d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max               �  50   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;      �  5�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3      �  6�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�ۄ   max       Pe�
      �  7�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��_o�    max       ?���`A�8      �  8`   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �D��   max       >E��      �  9,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�\(��   max       @F5\(�     �  9�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���R    max       @v���Q�     �  A�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @Q`           h  I�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�o           �  JP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Au   max         Au      �  K   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���n/   max       ?�͞��%�     �  K�   
         (   r            3               
      
         ]         #   3   F   =   I      9      	               \   *   +            	            	   t               N|&�NN�O.K�P9��Pf��Oh� N#3O
`@P�e�O�j�P���O�̰O�2M�ۄO|4KO"!�Om�N�gPK�XN�CIO"}�N�HPv1�Pf�PCO�WO;�O��uN��-N�jDNȶ�O��O�O-�P9lcO�tO5�0O(�IOj �N�;Nb2JM��'O�l/OQ��N:�O��yNBʤO��N��N�!tNz�׼D���o%   <o<#�
<49X<49X<D��<T��<e`B<u<u<�C�<�t�<�1<�9X<ě�<�`B<��=C�=\)=t�=�P=��=#�
=#�
='�=,1=,1=0 �=0 �=<j=<j=<j=<j=<j=D��=H�9=H�9=T��=Y�=Y�=aG�=aG�=e`B=�7L=�O�=�O�=�hs=���=�9Xkjnt�����tkkkkkkkkkk]anz~znia]]]]]]]]]]4/.15BN[^adc][NDB544���������
�����������)5BKKH=)��������������

�����

!#$)#
 �������NILN[gtu����{tg[WNNN����� )5CA)��������������������������	)5Ng������u[N	�����)8>ACA:6�������������������������������������������z}����������������������#),.) ��#)-6:BMORUVWOB62)##,./3<HUWZURUVUQH<4/,�����#/<?,66.�����`hhqt������|th``````���������������������������������������6EIVXXTLB5)��5Ng�������zn[5)����
#/:CU^H/#���������
#/8>A@<:/#
������������������NPWht�������tl[URQRN)*16COUZZOC6* ����

���������

���������������������������	
"
������
 )-,(#
��|�������������|��������������������������
"
����)-6;?FIFB6)!������##"�����.26BDOW[ag\[OB@6....����������������������������������������������������������������� %)%(����������������������������(142-�����
#'#
����������������������������������������"#/;@DD>;/#"

�a�n�z�~�|�z�n�a�W�V�a�a�a�a�a�a�a�a�a�a�<�E�@�>�<�/�)�*�/�6�<�<�<�<�<�<�<�<�<�<�n�{ŇŔŚŠşŗŔŇ�{�n�b�]�V�b�d�m�n�n�ûܻ��4�I�O�@�'�л����u�`�T�x����������0�<�R�[�Z�O�I�<�#�
��������Ŀ��������<�H�U�a�n�zÆ�z�x�zÁÂ�z�n�a�U�<�4�4�<�����������	���������������������������������������������������5�@�Q�Q�H�(�����Ŀ��������Ŀ�������(�5�;�D�7�5�0�(�������������)�B�[āēęĜĖč�t�[�6�������������"�.�;�\�f�g�`�G�;�.�"��	���������������������������������q�f�_�a�i�s��6�7�;�6�-�+�)�)�%�&�)�3�6�6�6�6�6�6�6�6�4�A�M�N�M�A�4������ݽݽ������(�4��������������������������������������ž��������ǾʾѾʾȾ�����������������������������������������������������������g¦²­« ¤�g�B�)����5�N�g�-�8�:�F�K�O�K�F�:�4�/�-�*�*�-�-�-�-�-�-�5�A�G�N�N�U�Y�Z�_�Z�N�F�A�3�(�&�#�(�0�5DoD{D�D�D�D�D�D�D�D�D�D{DoDkDbD^DbDlDoDoƧ�����$�@�I�=�$������ƧƎƁ�u�o�o�xƎƧ�	�"�/�E�O�K�4� �����������������������	���	��/�a�r�x�w�k�H�/�	����������������D�D�EEEE"E%E&EEED�D�D�D�D�D�D�D�D��������(�.�3�(������ݽ۽սݽ��Y�f�r�������r�Y�@�4�������'�4�M�Y�y�������������������y�m�e�b�e�g�m�o�y�y�Z�f�i�s�����������s�f�c�Z�T�Q�Z�Z�Z�Z�����ʾо׾���׾Ѿʾ�����������������������������������ŹŧœŒŔŠŭŴſ���Ҿ׾��	��"�.�/�.�,�"��	�����׾ξɾ׿`�m�y�������y�m�`�T�G�<�;�2�;�G�K�T�^�`�4�X�f�����ּ����ʼ�������r�M�>�4�%�4FF1FJFcFoFF�F}FoFcFJF=F1F$FFFE�FFEuE�E�E�E�E�E�E�E�E�E�E�E�EuEiEcEfEiElEu�F�O�F�>�:�4�-�!��������������!�3�F�������������������������y�l�h�e�h�y�������
�����#�$�#��
�	��������������������������������������������������������������������������������Óàù����������������ùàÓÇ��|ÁÉÓ�������
����������¿²¦²·¿������4�A�M�P�P�M�A�4�0�+�4�4�4�4�4�4�4�4�4�4�~���������źº��������~�r�L�A�:�B�U�r�~ǮǲǯǭǨǡǛǖǙǡǢǮǮǮǮǮǮǮǮǮ�	�
��"�%�-�.�"��	������������������	�l�x�������������x�q�l�c�l�l�l�l�l�l�l�l�����
�
���
���������������������������6�C�O�\�_�\�[�O�C�6�*�#�*�3�6�6�6�6�6�6  g % p  7 � = ) . %   3 � L F 2 7 " m i O N V D * O J Y U @ R r G { _ 0 g  � A n ' | 5 % b 7 E  \    ~  V  j  �  �  �  �  A  *    �  �  `  �  /  q  7  �  Z  �  �    ]  	  �  !  e  �  E  �  �  �  �  �  >  �  �  �  �  �  v  :  v  .  \  8  �  Z  3  �  ��o;��
<u=D��=��m=\)<e`B=\)=�%<��>�z�=@�=8Q�<�/='�=o=t�=8Q�=��m=�w=D��=�hs=�-=�/=��=�l�=�+=���=L��=T��=D��=��P=ix�=�C�>O�=�9X=�^5=�%=�o=m�h=}�=q��=���=��P=�o>;dZ=��=�1=��-=�-=���B
B�CB�B#+=B�]BD!B	�B	9�BT	B�MB��Bf�B@�B�B�xB��B�B�=B<�B��B �B�B��B�GB$B�B#U�BeB0Y�B$,�B$O�B��B�lB�BՖB^$B��BJsB.�Bw]B�IB�B9<B��B�B�B��B��B,�A���B*�B
!IBD7B?�B#=kB5�BA�B�`B	>�B�B��B�YB@5BAcB�(B��Bc�B=�B#�B=�B��B<NB��B�B	:�B�#B��B#@1BA�B0�+B$@�B$A�BA|B��B?B?�BȜB�B�]B-�B:�B��B9�B��BA�B�BI�BڦB��B,�A�B@)A�4�A�I|A��+@���A�hA�iA��A���A�>�A��AًAa$�AF @A�>�A6�1A��0AL0�A���A��j@}+mA��aC��)B��A��QA�9�C�N0A0�.@��AlϿAB"<AO\�A���AY�qAi�@���C���C�@k��A��A�ȜA�O�@M��A̕�A��IA:�Y@̶B�A�h�@�n8A��B ȧA�rSAÀ�A�T@��CA�8A�|FA�}�A��NA�\�A�8�AوKAafAF��A�Y�A7h'A���AL�A�z}A�Sc@��A�s�C��sB�A��A�yC�HFA0��@��Am�AA&lAM�A�{eAW�Ai6p@� �=��DC��@`EA��A���A��@L�ÄaA�GlA:��@qB�A�]�@��wA�kB �&            )   r            3        !      
               ]         $   3   G   >   J      :   	   
               ]   +   ,            
            	   u                           ;   /            5      ;   #                     1            5   7   1         #            !         9   #                              %                                          '                              !            3   !   -                     !         %                                                N|&�NN�O.K�N�	O��#Nr(�N#3N�gP"�FOi�O]#yO�ɻOj),M�ۄO/q�O"!�Om�Nq-�O���Nz%O�N�3
Pe�
O�o�P']O8̺N�O[��N��-N\�=Nȶ�O��O�N��HO���O��VO,2O(�IOj �N�;Nb2JM��'O�l/OQ��N:�O��NBʤOAON��N�!tNz��  �      �  
�    �  !  �  t  �  �    #  �  <  �  )  �  1  U  g  �  �  �    �  �  {  8    `  �  W  
)  %  �  �  �  �  �    �  �    \  �  ?  *  �  ��D���o%   =\)=m�h<�9X<49X<�t�<���<�o>E��<�1<�9X<�t�<���<�9X<ě�=+=�7L=t�=t�=��=�w=e`B=8Q�=e`B=49X=q��=,1=49X=0 �=<j=<j=P�`=��=P�`=P�`=H�9=H�9=T��=Y�=Y�=aG�=aG�=e`B=��=�O�=�\)=�hs=���=�9Xkjnt�����tkkkkkkkkkk]anz~znia]]]]]]]]]]4/.15BN[^adc][NDB544����������  �������������).6;95)�������


������������

!#$)#
 �������RQY[egt~}utg^[RRRRRR������� "�������������������������A<;<@BN[gpwxvtrg[NBA������)3;<96)��������������������������������������������������������������������#),.) ��#)-6:BMORUVWOB62)##./29<BHRRHA<:/......������
"# 
������ntt������tnnnnnnnnnn���������������������������������������(5BGTWWSJ5) � $5N[gt�����tgNB5)������
#/8BMMH/#����
#$/29<;0/#
��������  �������YWXYX\aht������tg[Y)*16COUZZOC6* ��� 

�����������

���������������������������	
"
�����	
#%)'##
 �������������������������������������������
!
���)-6;?FIFB6)!������##"�����.26BDOW[ag\[OB@6....����������������������������������������������������������������� %)%(�����������������������������$$"����
#'#
��������������������������������������"#/;@DD>;/#"

�a�n�z�~�|�z�n�a�W�V�a�a�a�a�a�a�a�a�a�a�<�E�@�>�<�/�)�*�/�6�<�<�<�<�<�<�<�<�<�<�n�{ŇŔŚŠşŗŔŇ�{�n�b�]�V�b�d�m�n�n�������ûлӻۻлûû����������������������
��#�0�<�<�:�5�0�#�
�����������������U�a�n�n�r�o�n�a�U�S�H�T�U�U�U�U�U�U�U�U�����������	����������������������������������������������������(�5�C�D�?�5�����ݿѿĿ����ÿѿ�����(�3�=�5�2�-�(���� �����������6�B�O�[�h�k�t�u�p�h�[�O�B�:�6�-�+�-�4�6��"�.�;�G�Q�_�a�^�V�G�.�"������
��s���������������������s�g�f�d�e�j�n�s�6�7�;�6�-�+�)�)�%�&�)�3�6�6�6�6�6�6�6�6��(�4�A�H�J�I�A�@�4�(������������������������������������������������ž��������ǾʾѾʾȾ����������������������������������������������������������[�g�t�w�g�[�N�B�5�(�(�,�5�B�N�[�:�F�G�K�H�F�:�7�4�/�:�:�:�:�:�:�:�:�:�:�5�A�D�M�N�T�X�Y�Z�N�H�A�5�4�(�'�$�(�3�5D{D�D�D�D�D�D�D�D�D�D{DoDlDbDbDbDoDwD{D{Ƨ�����$�0�=�E�<�$������ƧƎ�w�q�q�zƎƧ�	��"�4�D�D�A�9�"�	�������������������	��������/�H�a�o�u�t�h�H�/��	����������D�D�EEEEE EEEEED�D�D�D�D�D�D�D�������(�+�(�������߽ݽٽݽ����'�4�@�M�Y�f�r�{�|�r�f�Y�M�@�4�"���"�'�y�������������������y�m�e�b�e�g�m�o�y�y�Z�`�f�s�����|�s�f�d�Z�U�U�Z�Z�Z�Z�Z�Z�����ʾо׾���׾Ѿʾ�����������������������������������ŹŧœŒŔŠŭŴſ���Ҿ׾��	��"�.�/�.�,�"��	�����׾ξɾ׿m�y�������}�y�o�m�`�T�G�F�D�G�Q�T�`�d�m�r�������ʼּ����ּʼ�����������f�rFFF1FJFVFfFoF|FFzFoFcFJF=F5F"FFFFE�E�E�E�E�E�E�E�E�E�E�E�E�EuEkEiEeEiEuE��F�O�F�>�:�4�-�!��������������!�3�F�������������������������y�l�h�e�h�y�������
�����#�$�#��
�	��������������������������������������������������������������������������������Óàù����������������ùàÓÇ��|ÁÉÓ�������
����������¿²¦²·¿������4�A�M�P�P�M�A�4�0�+�4�4�4�4�4�4�4�4�4�4�r�~���������������������~�r�e�Z�R�U�e�rǮǲǯǭǨǡǛǖǙǡǢǮǮǮǮǮǮǮǮǮ���	��"�$�,�-�"��	��������������������l�x�������������x�q�l�c�l�l�l�l�l�l�l�l�����
�
���
���������������������������6�C�O�\�_�\�[�O�C�6�*�#�*�3�6�6�6�6�6�6  g % J  / � 5 * ,   . � 0 F 2 *  _ W O M 5 C " L D Y S @ R r H Y _ 3 g  � A n ' | 5  b 8 E  \    ~  V  j  �  �  s  �  �  �  �  �  ^  �  �  q  q  7  {  i  c  c  �  8  6    �  "  �  E  �  �  �  �    M  Q  `  �  �  �  v  :  v  .  \    �  2  3  �  �  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  Au  �  �  �  y  h  X  G  5  "    �  �  �  �  �  y  O     
  �    "  ,  4  :  ?  H  T  `  w  �  �  �  �  �  �  	    /  C    �  �  �  �  �  �  �  �  �  �  �  c  A    �  �  R    �  2  G  R  K  P  M  S  K  B  D  C  C  8  �  l  !  �  
  {  �  I  �  	|  	�  
6  
p  
�  
�  
�  
�  
�  
�  
1  	�  	4  �  �  �  �  �  s  r  p  {  �  �  �  �  �      !  	  �  �  W    �  K  �  �  �  �  �  �  �  �  �  �  �  v  f  V  F  3  !     �   �   �  �  �                 �  �  �  �  Z    �  �  *  o    &  @  Z  �  �  �  �  �  �  �  x  g  L    �  �    �  .  �  f  o  s  q  m  e  X  I  <  0  %      �  �  �  �  x  =  �    5  �  v  �  y  �  j  �  x  �  K  �  l  �  �    �  
�    �  �  �  �  �  �  �  �  �  �  �  �  V  )  �  �  m  �  t  t  �  �             �  �  �  �  [    �  �  $  �  l    �  #  &  (  *  ,  .  1  >  P  i  �  �  #  w  �  �  �  �  �  �  0  v  }  �  �  {  t  j  [  K  8  "    �  �  �  Q  !    5  <  *        �  �  �  �  �  r  R  3    �  �  �  �  ,  �  �  �  �  �  �  �  �  �  �  �  r  ^  @    �  �  �  f  '   �    �  �    #  (  '      �  �  �  �  r  K  !  �  �  �  �    a  �  �    V  �  �  �  �  Y    �  o  /  �  y  �  W    1  .  ,  )  (  *  -  /  2  5  8  ;  3  #      �  �  �  �  L  R  F  )    �  �  �  �  \  3    �  �  a  *    �  �  K  9  d  I  $  �  �  �  Z    
�  
�  
,  	�  	x  	  �  N  �  �  �  �  �  �  �  �  �  �  �  �  P    �  �  D  �  �  =  �  G  �  �    .  f  �  �  �  �  �  W    �  �  �  ^    �  �  �  �  s  �  �  �  �  e  P  F  4  )    �  �  6  �  W  �  G  �  �  �  �  �        �  �  �  |  H  
�  
�  
)  	y  �  �  �    #  �  �  �  �  �  �  �  |  U  %  �  �  |  -  �  �    �  T  �  *  [  �  �  �  �  �  �  �  �  8  �  m    �  �    u  �    {  p  d  W  J  ;  *      �  �  �  �  �  {  f  Q  <  &    8  8  8  .  "    �  �  �  �  {  Z  8    �  �  �  �  �  l    �  �  �  �  �  �  �  �  �  �  �  �  k  P  5  &        `  U  I  8  !    �  �  �  �  q  @    �  y  %  �  {  �  �  �  �  �  �  �  �  r  V  6    �  �  �  W    �  r     �   X  !  H  Q  U  W  V  N  ?  *    �  �  �  ;  �  �  +  �  �  3  �  	u  	|  	u  
(  
  	�  	�  	�  	O  	E  	C  �  �  "  �  �  ^  �  �      %  "      �  �  �  �  �  =  �  k  �  �  `  �  Z  �  �  �  �  �  �  �  z  Z  3    �  �  j  �  e  �    X  c  $  �  �  t  d  k  W  ?  #    �  �  y  D  0    �  �  q  c  K  �  �  �  �  w  c  O  7    �  �  �  �  ]  +  �  �  G   �   _  �  �  �  �  �  �  �  ~  j  V  @  *    �  �  �  o  .   �   �  �  �  {  m  _  D  &      �  �  �  �  �  `  ?    �  �  �       �  �  �  �  �  �  k  N  +     �  �  |  O  !  �  �  �  �  z  a  H  %    �  �  �  �  �  _    �  Y  �  �  >  �  J  �  �  �  t  T  -    �  �  �  �  s  K    �  �  G  �  e  m    �  �  �  �  �  �  �  �  }  `  =    �  �  I  �  �  O   �  �  [  �    C  X  \  R  2  �  �  Q  �  `  �  
�  	�  h  �  }  �  �  �  z  h  V  E  4  #      �  �  �  �  �  �  �  �  �  )  :  9  .       �  �  �  �  �  y  ]  =    �  �  �  X  �  *  "      	     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  a  ;    �  �  �  �  �  i  I  )            !  +  �  �  �  k  X  K  8  #    �  �  �  m  G  "  �  �  �  x  H