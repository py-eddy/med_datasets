CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?ԛ��S��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Mޠ�   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �+   max       =�`B      �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?E�Q�   max       @F�\)     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @vd(�\     �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @O            t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�2        max       @�V�          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       >�V      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�.-   max       B,�J      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~   max       B,�       �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�.�   max       C��t      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��i   max       C���      �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max         C      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Mޠ�   max       Pb��      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��ᰉ�(   max       ?�\(��      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �+   max       >Kƨ      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?Tz�G�   max       @E��
=p�     �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����
=p    max       @vc��Q�     �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @O            t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�2        max       @�          �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         AE   max         AE      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��ߤ?�   max       ?�����C�     �  QX         	   '         7      j      )   ?  B   S         
   #      )                  !         !                  6            	   0   �         K   P   )            
   S               2   N�bO�sM�h�O��%N��NȽO���N0~�Pb��O	?P~�Pr.OP���P��NHC�O�v�N���P�Op�O�_N�(-N��$Nn8!N��*N���PÍO7�N�"O�R�N:��N�L�O�suN@!O7�Pr��NL�O�g�O���N��O$:�O�Q�O.��Nv�)P��O�ɎO���N��O��DOx�N�.5O؊QN�9NuI�O%�[Mޠ�OP�{N蛽+�T���ě���o��o��o;��
;��
<o<o<T��<�o<�o<�o<�o<�t�<�t�<���<��
<�9X<ě�<ě�<���<���<�/<�/<�/<�h<�h<��=t�=��=�w=,1=,1=,1=49X=D��=Y�=]/=aG�=aG�=e`B=e`B=m�h=u=y�#=y�#=}�=}�=��=���=�1=���=���=��`=�`B����������������������������������������������#065/#
�������fa`gnt�������ztgffff��������������������" /<HUanz|twniU</%"��������������������������)48::1)���.35BDN[`fgd[NB95....xw����������������x��������
���������	5N[��������tW){~���)482)�������{���������������������������
#%������#*05630#���������������������������������������������������������EOOZ[horqkhf[OEEEEEE�w{���������������������z}��������������FCKNR[^gigf_[NFFFFFF������so�����������������s��������������������<AGHOUalnpqneaVHC><<��������������������)()5:BFHB54)))))))))��������������������
'4<GONIH</#
DHHHDHT]\XTHDDDDDDDD7;AHTamvxvmaTH@;7777fgwvy������
����tf"#&/<EHSH</#""""""""�����
#/<EFF?<#
�����%)+1,) ��{�������������{{{{{{����������������������������

���������� %,/)#�������

���������ILLTW[htv��������tOIfb��������������zpnf��������*67-����������	��������$)5BMSWYVNHB5(!)5:@CCB75)%)+,-+)���������
������)+6BOROHB6)%.*'(0<BCE<;0........
#+/<>@F?</#
��������������������������

���������������������������I�K�V�[�a�V�I�=�0�'�+�0�=�D�I�I�I�I�I�I���������������������������~�~������������������úùôù�����������������������޻����������������_�F�8�/�/�7�S�_�l�x�����������������������������������������������������������������������������������������!�$�'�����������������������������ʼ˼̼ʼ����������������������������
�<�b�xŀ�}�p�b�I�0������������������n�{�}ŅŇŐœŇ�{�n�b�[�Y�\�b�e�n�n�n�n�Z�s������������g�Z�W�N�A�(�$��!�.�A�Z��������6�A�=�,�����ŭśŞůŷŲŤŜ���6�O�hčğĪĩĚ�t�O�<�6�)���������6�����U�^�[�H�/�	������������������������	���������������������������%�$����������ݿٿؿӿҿݿ��ּ��������ּʼ��üʼͼּּּּּ־�)�4�����齷���������z���������ӽ���)�6�=�B�O�^�e�c�[�B�6�,�#�)�6�9�6�)�'�)ìù��������� ���������ùìèßÝÙàì�H�U�U�a�b�k�a�U�H�<�;�<�>�>�H�H�H�H�H�H�/�<�H�K�S�T�H�@�<�/�/�/�,�&�'�/�/�/�/�/���������������ĿǿĿ¿������������������Ľнݽ������ݽսнĽ����ĽĽĽĽĽ��z�������������z�m�b�a�_�a�c�m�n�z�z�z�z��4�M�f�s�������s�f�Z�M�A�4�&������G�S�T�`�g�m�m�m�l�g�`�T�N�G�;�9�;�@�D�G��(�5�7�5�2�.�5�6�5�(�$���� �����N�Z�`�l�j�e�l�Z�N�A�5�.�(�#�"�(�5�9�A�N�(�4�>�9�4�+�(�����(�(�(�(�(�(�(�(�(���������� ��������ܻл˻˻лܻ�������������������s�f�Z�V�T�X�[�Z�f��U�`�b�e�n�q�u�n�b�U�M�L�U�U�U�U�U�U�U�UĚěĦīĭĲĮĦĚčąćĈċčĖĚĚĚĚ�������	��/�;�H�S�K�<�"�����������������"�$�&�"�!��������������[�g�t�x�t�s�g�[�N�B�>�-�)�/�B�[�;�G�T�`�b�e�b�T�H�;�"�������	��"�.�;�
��������
� ���������
�
�
�
�
�
Óàì÷ù����úùìàÞÓÐÇÄÀÃÇÓD�D�D�D�D�D�D�D�D�D�D�D�D�D�DDsDsD|D�D��F�S�_�f�l�s�k�X�S�F�:�-�$�!��%�-�:�B�F�"�.�;�;�<�;�.�"�����"�"�"�"�"�"�"�"���4�M�r���������|�Y�M�@����������ɻ����������ֺ����������������y���������������������y�l�`�[�]�\�_�_�y������������������������ƴƺ�������������������$�-�9�;�0�$�������������������N�[�g�t�}�t�g�[�B�5�)�(�(�.�5�;�N¿��������������¿²¦¦²»¿¿���ùϹܹ������������ù����������'�3�7�?�3�*�2�0�'� �������'�'�'�'���ɺֺ����ֺɺ����������������������6�@�C�K�C�=�6�/�+�*���������(�6����������������������������������������EvE�E�E�E�E�E�E�E�E�EuEiEfEdEeEdEcEiEtEv�нӽݽ��������ݽнʽʽннннннн� ?  t - ? [ - L 6 2 C : ) @ - H - T F ) T W T 3 c > : C E ^ 7 J V P G i , V i  & J F J R j ! 6 8 u D s 1 M | 1 r    �  )  Q  �  �  M  �  d  �  >  �  �     &  g  �  �  �  0  n  �  
  �  �  �  r  )  *  z  V  �  O  N  \  -  �  k  8  �  `  h  �    �  \  I    Q        )    �  �    �  M����<49X;D��=\);�`B;ě�=u<o=�l�<���=Y�=��w>�V=ȴ9<�t�=<j<�/=Y�=H�9=}�=\)=C�<�`B<�/=o=q��=C�=�P=y�#=+=@�=�%=0 �=aG�=\=D��=��=�o=y�#=��`>Q�=���=y�#>$�>I�=��=��=�9X=��
=�hs>�w=�-=ě�=�G�=�
=>��=�FB�\B!КB��B$cB	�[B�0B �B#"B�kB"�B�B��Bi�B�6B+�PBX�B%/:B!{B��B�~BC�Bx*Bl�B��B�BMBQ�B��B�uBS�Bj�B�~A�.-A�PB�B�-B>BvGB>BB"*B��BB�B��B3�B��B,�JB��Bz�B�'B4�B=B��B%�SB�NB)BW)B��B��B!��B?�B$BqB
J�B��B��B#)�B��BcBC�B�SBDB�B+��BK$B%?�B!��B�yB�[BAnB�hBqFB�JB�9B�B@�BX�B �BC>BL=B�kA�~A�~�B9�B�CBRFB!mB{`B"@'B��B��B��B>�B�B,� BA�Bj�B��B@�B�B�?B%�B}SB;�B��B�B�@�A��@��"A��CA��4AҜE@�NSA�|(A��7A��A���A�DZA�[�@VzA��HA��A)�JA�%A�:4A�6A�@�Au}A)�zA��vA<�Af�A�U�A�A7a/@�$�AEqA�UAߋ{A��UA�v�A�NzAa0�A�-A˛pC��z@�@$A_�(@ͫ�@C:KA��B�*B��A�GA���>�.�?�@@9�HA��@�`�C��tA+��B
�,@�A�x[@��IA�h�AЃ�AҀ@�:A�%A�yA�o�A��SA�ZA���@[�A��A�A+@�A�y�A���A�}�A��Av5�A)�
A�g�A<خAe��A��A��#A73@��1AF�lAA߇�A��yA��A� �AbI�A���A�jC��z@�J�A`��@˙�@DJ�A B��B�A�;�A��(>��i?�/x@<COA���@�ߠC���A*��   	      
   (   	      8      j      )   ?  C   S         
   #      )                  !      	   !                  6            	   1   �         L   P   *            
   T               3               !         !      /      +   5   ;   =            +                        '                           5                           +   #   #               !                                                   !   !      3                                                                                          '                                       N�bN�ƳM�h�O�LSN��NȽN���N0~�O��@O	?O�z�O�O�*�Pb��NHC�O�]�N��=O���N6��ObݕN��{N��%Nn8!N��*N���O��O7�N�m\O:PON:��N�L�O�0pN@!N���O� �NL�O|��Om��N��O$:�O-LO.��Nv�)O�E�O�Z5NѾBN��O\>Ox�N�.5O�q�N�O�NuI�O%�[Mޠ�O�N�  /  �  �  O  �     �  u  
~  s  ]  ~  �  �   �  �      z  �  >  �  �  U  �  1  E  �  �  �  �  >    �  �  �    5  �  	@  �  �  �  3  �  $  �  4  �  ~  �    �  c  <  
7  ��+�49X�ě�;�o��o��o=+;��
=D��<o<�j=��>Kƨ=C�<�o<�j<���<��=o=+<���<���<���<���<�/=\)<�/<�=��<��=t�=�w=�w=0 �=�o=,1=H�9=H�9=Y�=]/=�h=aG�=e`B=�7L=���=��
=y�#=�+=}�=}�=�1=��-=�1=���=���=�S�=�`B����������������������������������������������
#+1/+%#
����fa`gnt�������ztgffff��������������������../6<CHSUZUTH<1/....�������������������������)./-)����.35BDN[`fgd[NB95....���������������������������������������-,-5BN[gmuxwtjg[NB5-�����+0/)����������������������������������
�������#(04510#������������������������������������������������������������GOP[[hmpoh[OGGGGGGGG}�������������}}}}}}����z}��������������FCKNR[^gigf_[NFFFFFF����������������������������������������������E?<CHUafhdaUQHEEEEEE��������������������)()5:BFHB54)))))))))��������������������	'3<FMMHF</#DHHHDHT]\XTHDDDDDDDDF=CHTamuwtmaTHFFFFFF��������������������"#&/<EHSH</#""""""""���
#/<ACB<:/#
�����#(*/.({�������������{{{{{{����������������������������	

���������� %,/)#�������

���������SOOPY\ht~��������t\S}{{����������������}�������������������������	��������$!!"')05BHNTUSMIB5+$!)5:@CCB75)%)+,-+)�����������������)+6BOQOFB6)'.*'(0<BCE<;0........
#+/<>@F?</#
���������������������������

��������������������������I�K�V�[�a�V�I�=�0�'�+�0�=�D�I�I�I�I�I�I������������������������������������������������úùôù�����������������������޻x�����������������l�S�F�D�>�9�F�I�S�_�x�����������������������������������������������������������������������������������������������������������������������ʼ˼̼ʼ���������������������������#�0�<�J�O�L�C�1��
������������������n�{�}ŅŇŐœŇ�{�n�b�[�Y�\�b�e�n�n�n�n�N�Z�s����������p�g�Z�N�A�5�'�*�.�6�A�N�������*�1�1�*��������ŶŭŰŸ�������O�[�h�t�|ā��y�t�h�[�O�B�6�4�1�2�6�@�O�	�"�<�L�U�O�D�/�	���������������������	��	��������������������������������	�����޿ܿ޿ۿݿ���ּ�������ּʼżżʼϼּּּּּֽݽ�����������߽ڽĽ��������������B�O�V�[�T�O�B�?�9�@�B�B�B�B�B�B�B�B�B�Bìù����������������������ùóìçæçì�H�R�U�a�b�i�a�U�H�>�@�@�H�H�H�H�H�H�H�H�<�E�H�O�M�H�>�<�/�-�(�)�/�6�<�<�<�<�<�<���������������ĿǿĿ¿������������������Ľнݽ������ݽսнĽ����ĽĽĽĽĽ��z�������������z�m�b�a�_�a�c�m�n�z�z�z�z�(�4�A�M�Z�f�s�x�}�|�s�f�Z�A�4�)����(�G�S�T�`�g�m�m�m�l�g�`�T�N�G�;�9�;�@�D�G����(�5�/�)�(���
����������A�N�Z�`�`�`�b�c�Z�N�A�5�0�(�'�(�-�5�=�A�(�4�>�9�4�+�(�����(�(�(�(�(�(�(�(�(���������� ��������ܻл˻˻лܻ�������������������s�f�\�Z�V�Z�\�\�f��U�`�b�e�n�q�u�n�b�U�M�L�U�U�U�U�U�U�U�UčĚĦĩĬıĭĦĚčĆĈĉččččččč���	��/�3�9�7�4�.�"��	�������������������"�$�&�"�!��������������[�g�t�{�s�p�n�g�[�N�D�B�:�3�.�5�B�K�[�"�.�;�G�`�a�c�a�T�G�;�"�������� �	��"�
��������
� ���������
�
�
�
�
�
Óàì÷ù����úùìàÞÓÐÇÄÀÃÇÓD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��F�S�_�f�l�s�k�X�S�F�:�-�$�!��%�-�:�B�F�"�.�;�;�<�;�.�"�����"�"�"�"�"�"�"�"���'�4�M�r����r�Y�M�@�4������������ɺֺ�������������ֺĺ��������y�����������������y�q�l�l�l�n�q�y�y�y�y������������������������ƴƺ�������������������$�(�0�5�6�0�&�����������������N�[�g�t�}�t�g�[�B�5�)�(�(�.�5�;�N¿��������������¿²¦¦²»¿¿�ùϹܹ����
�������ܹù����������ú'�3�6�=�3�)�1�.�'�"������ �'�'�'�'���ɺֺ����ֺɺ����������������������6�@�C�K�C�=�6�/�+�*���������(�6����������������������������������������EuE�E�E�E�E�E�E�E�E�E�EuEjEiEfEiEiEkEsEu�нӽݽ��������ݽнʽʽннннннн� ?  t * ? [  L & 2 5 8  0 - : * H ) ) S E T 3 c  : ' 6 ^ 7 J V G ! i - P i   J F A D F ! 6 8 u @ v 1 M | ) r    �  �  Q  6  �  M  �  d  �  >  �  %    �  g    �  F  S  �  �  �  �  �  �  I  )  �  �  V  �  6  N    �  �  �     �  `  J  �    '  6  �    �        �  �  �  �    D  M  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  AE  /  .  .  ,  *  &               �  �  �  �  \  &  �  �  �  �  �  �  �  �  �  �  �  }  n  [  :    �  �  Y    �  �  �  �  �  �  �    p  j  h  e  c  (  �  F  0      �  �  �  8  E  M  N  N  C  2  $        �  �  �  s  2      ^   �  �  �  �  �  }  m  U  =  0  %        �  �  �  �  �  �  �     !  "  "            �  �  �  �  �  �  �  �  |  k  Z  9    c  �  �    C  t  �  �  �  �  u  (  �  Y  �    d  "  u  u  u  u  t  t  t  o  i  b  [  U  N  K  Q  V  \  a  f  l  �  ^  	  	�  
  
K  
m  
}  
z  
b  
3  	�  	�  	  m  �  �  �  o  B  s  g  [  P  E  :  6  2  .  )       �  �  �  �  |  O  �  \  �  �  ,  I  Y  \  O  9    �  �  �  �  {  6  �  �  6  �    �  	  ]  �     T  x  }  t  Y  0  �  �  %  �  ;  �  �  �  	  �  �    $  �  z  �  �  �  "  w  �  Z  �  �  �  �  <  �  C  �  '  Z  t  �  z  Z  $  �  �  S  #  �  �  f  �  r  �  �  v   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �  }  �  �  �  �  �  r  [  H  4      �  �  �  L  �  �  �   �                �  �  �  �  �  �  g  A    �  �  �  �  �  �  �  �  �  �       �  �  �  �  �  h  8    �  r     �    +  �  �  '  <  B  W  |  �  �  �  x  Z  3  �  �    w  �  #  [  �  �  �  �  �  �  �  {  K    �  V  �  i  �  �    �  9  <  =  :  6  2  .  *  *  $    �  �  �  �  x  Q  '  �  �  �  �  �  �  �  �  �  �  �  �  o  Z  B  )    �  �  �  �  t  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  U  T  R  Q  P  N  M  K  J  I  H  I  I  I  I  J  J  J  K  K  �  �  �  �  �  �  �  �  �  �  �  {  u  q  m  i  �  �  �  �  �  �  �    .  0  (  !         �  �  �  z  :  �  �  k  c  E  9  .  #    	  �  �  �  �  �  �  �  �  �  w  [  ;     �  �  �  �  �  �  �  �  �  �  w  b  K  5      �  �  �  X  	    O  {  �  �  �  �  �  �  �  l  N  "  �  �  /  �  *  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  i  \  O  �  �  �  �  �  �  �  �  j  S  <  $    �  �  �  �  a  �  k  4  >  9  -      �  �  �  �  \  4    �  �  e    �  =  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �    h  R  ;  $  �  �  �  �  �  �  �  �  �  r  X  :    �  �  �  �  \  ,  �    x  R  v  �  �  �  �  �  �  �  j  >    �  �  2  �     �  �  �  �  |  j  W  D  /      �  �  �  �  �  p  W  <  !                  �  �  �  �  l  4  �  �  g    �  T  �  /  4  3  -  "      �  �  �  �  �  �  u  P    �  t     �  �  �  �  �  �  �  �  �  �  w  ^  D  *    �  �  �  �  z  X  	@  	+  	  	  �  �  �  �  �  U    �  Y  �  %  �  �  +  v  �  d  �  q  �  %  g  �  �  �  �  �  R  �  �  �  �  %  �  �    �  �  {  q  b  N  5    �  �  �    Q  !  �  �  \    �  `  �  �  �  �  �  �  u  f  Y  N  C  8  6  9  <  ?  E  L  R  Y  �    ,  1    �  �  �  �  �  �  �  �  >  �  �  (  ~  �  8  
�  R  �  �  �  �  �  �  x  D  
�  
�  
7  	�  	%  b  �  �  �  {    �  �  �  �  �  �      $         �  �  i  �  �  �   �  �  �  �  �  �  �  }  s  h  `  W  M  <  '    �  �  �  n  C  !  -  1  4  4  2  /  %      �  �  �  u  @    �  w  '  �  �  �  �  �  �  }  `  C  $    �  �  �  s  F    �  �  G  9  ~  y  s  k  c  W  K  >  1  (  "    �  �  �  �  `  &  �  �  �  �  �  �  �  �  �  �  Q  
�  
�  
K  	�  	p  �      �  �  Q          �  �  �  �  �  �  �  n  C    �  �  �  E  /  K  �  �  �  �  u  X  =  '    �  �  �  �  �  n  D    �  >  �  c  P  <  %    �  �  �  �  �  �  �  �  �  �  �  |  h  X  I  <  *      �  �  �  �  �  �  �  �  �     �  �  �  �  �  �  	�  
  
-  
5  
6  
/  
  	�  	�  	z  	5  �  n  �  ^  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  r  f  T  A  .      �  �  �