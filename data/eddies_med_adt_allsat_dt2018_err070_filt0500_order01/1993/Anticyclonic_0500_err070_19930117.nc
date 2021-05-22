CDF       
      obs    ;   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��t�j      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =�v�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>\(��   max       @F�����     	8   p   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��         max       @vP          	8  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @N�           x  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�&        max       @�``          �  3X   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �\)   max       >I�      �  4D   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�Ks   max       B/n�      �  50   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�|t   max       B/n�      �  6   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =���   max       C�C�      �  7   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =�G�   max       C�Mz      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  8�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7      �  9�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          /      �  :�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P:��      �  ;�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�쿱[W?   max       ?�b��}Vm      �  <�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       =�v�      �  =|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���R   max       @F�����     	8  >h   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @vNfffff     	8  G�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @M�           x  P�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�&        max       @��          �  QP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =j   max         =j      �  R<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�e+��a   max       ?�]�c�A!     p  S(            �   
         7      2                        (   ;   
   6                     A      k   $      &   +      ,         !             \   &                  )         0   B   	            N�N&RN�&.PK2�N�;�N
��Nn7P�N7,hO�y�N���O���N��P��O*VlO��N�v�O�NQP|�6N�d�O���O'�+N�4�M��rOL�hOTGMN��P1`Ozn�P��O�ntNQ�P#d�O��N��P%Nw�O2��O]g�M��O�3�N�VuP/��OkNFN�N�AWN�t�N��OX�_P#�OtpNǦ&O��O��N6��Nђ0N��Ni��NR﫽���󶼴9X���㼓t��D���49X�ě��ě����
;�o;��
;��
;�`B<t�<#�
<#�
<49X<D��<D��<e`B<�C�<�C�<�t�<�9X<�j<ě�<���<���<���<�/<�`B<��<��<��<��=C�=C�=C�=C�=��=��=#�
=#�
='�=,1=0 �=<j=@�=T��=Y�=aG�=e`B=q��=}�=�o=�7L=�C�=�v���������������������#09<=<60,# -*+8?Ngt�����tg[NB;-Z[gt�������tgb\[ZZZZ,.0<INIG<0,,,,,,,,,,���

���������}{�����������������}%#)6>@<60)%%%%%%%%%%LJLN[g���������tg_WL���������������������������������������������������������.4--8AWansot~�yraU<.����������������������
#%(0300*#
	(*68:76*�������������������������
#/HUaknhH<
���@BEGO[]dhhjhb[WOKB@@�����*.00.&������������	�����������������������{������������������)-59CDD=5/)�����

��������������5=FHKLHB5����SUaz���������zna]UUS`afnw������������ze`<HQQLKE;/""%/<@?ABGOZZ[\[OGB@@@@@@)5Ohyz���g_NE5&57;BN[got���ugNB<65�����������������������6>GNL6-)�����6)$)+686666666f_chint���������tohf��������������������RMRUbdebZURRRRRRRRRR(#%(/<HRUVVTOHH=<8/(*)+/<HOUZUTHD<5/****���5FZ_a^VB5)
�  $/<HMMI</#��������������������xvz��������������������������������������������������	
#/<DBDB<1/#
	������)--'�����\UUX]_gmz������zsma\�����������������������������������z������������������z)'),169;<;:62)))))))��������������������AEHLTTT\YTNHAAAAAAAAz{����������zzzzzzzz�

���������������������������������������������������������������������������������������������һ������������������x�n�l�a�l�x�{����������)�6�h�t�x�w�q�P�B�6��������������������������������������������������������'�4�6�:�?�4�(�'�$�%�'�'�'�'�'�'�'�'�'�'�m�y���������y�m�`�Y�`�h�m�m�m�m�m�m�m�m�A�Z�������������A���ݿѿֿ�����)�AÇÓÖÞÓÇ�z�p�zÁÇÇÇÇÇÇÇÇÇÇÇÓàâååíçàÓÇ�n�a�[�W�V�Z�a�nÇĦĳĸĿ����ĿĳĳĦĞĚĕĘęęĚģĦĦ�/�;�H�L�a�j�j�a�V�H�"���������	���/�#�/�;�<�D�?�<�6�/�)�#���"�#�#�#�#�#�#�a�m�������������������������z�a�L�J�Z�aĦĳĿ������������������ĿĳĦĤĢġĥĦ�4�A�M�Z�f�o�q�j�f�Z�M�A�4�1�(�%�&�(�0�4�T�`�m�n�s�m�l�a�`�T�G�F�A�?�G�L�T�T�T�T�Ϲ����	������Ϲ��������r�e�k�x���Ͽ����5�A�e�q�s�l�M�6�(���ݿտ��������������������������y�s�l�s�|��������@�M�Y�\�c�d�o�r�f�Y�@�4�'������'�@����&�*�6�=�C�J�C�6�.������������ìù��������ÿùìàÛààäìììììì�����������������������������������������;�G�L�T�T�S�H�G�;�.�"��
��	���"�.�;�T�`�m�y�����������y�m�`�T�G�>�;�5�7�G�T�a�n�y�s�u�w�n�a�]�U�J�N�U�\�a�a�a�a�a�a��%�0�<�k�{Ń�{�b�I�0��
��������
�����
��������������������������������������û˻ܻ���������л��������������������m�a�H�;�/����"�/�H�T�a�m�y�z�����лܻ����������ܻۻջл̻ллллл������������������������q�j�s�w��������������0�A�K�A�>�(����ݿԿ˿˿ѿݿ������������������������������������������/�;�D�D�A�1�"�	���������������������	�/�Z�S�Q�Z�g�s�������z�s�i�g�Z�Z�Z�Z�Z�Z�Z��'�4�?�@�M�W�W�M�@�4�'�����	������'�+�3�4�'�!������������������������������������������������������ʾ�	��"�+�+�"������׾ʾ�������������������	����������������������������B�N�[�w²¸¥�t�g�[�B�!����)�BD�D�D�D�D�D�D�EEEE$EED�D�D�D�D�D�D������ùŹɹù����������������������������������ĿʿĿ���������������������������������������������������ſž�����������ƽ��Ľ˽˽ŽĽ����������������������������B�N�[�g�l�p�s�s�g�[�N�B�7�)�'�#�)�5�>�B������-�:�]�l�l�F�-������Ⱥĺպ���{ŇŔŠŭŹ����������ŹŭŠŔŀ�{�u�t�{����(�4�8�A�L�O�M�A�4�(�$���������������������������������z�n�k�w�|���������������ּ�������ּʼ����������������e�r�z�~�������~�r�e�Y�W�Y�e�e�e�e�e�e�e�l�x���������������������x�l�h�e�l�l�l�l�
���#�0�0�0�#���
��
�
�
�
�
�
�
�
����������������������������������������ǭǯǭǡǔǋǏǔǞǡǢǭǭǭǭǭǭǭǭǭ d O 5 $ @ f 6 t ; 0 U % T H . ' 7 j 1 < A H = H   2 B , H A + I O 7 ? ; E  2 = ^ ! 6 y > A @ F ! m : u N z k S q w ?    E  X  �  ]  >  P  w  {  J  �      �  �  q  V  �  v     �  �  {  �    �  �  �  )    �  �  s      �  �  �  o  �    �  �  �  @  *  �  �  B  �  �  �    �  �  |    l  �  b�\)��/�u>I��t��t����
=H�9$�  =8Q�<�t�<���<T��<��<��<�1<T��=P�`=�hs<�1=�C�=+=C�<�9X=�w=t�=\)=�E�=49X>+=�%=\)=�O�=��P=�P=���='�=}�=�+=��=�\)=}�>$�=��-=T��=H�9=e`B=D��=��-=�v�=���=�C�=���=��=�hs=��-=��=���=��B!��B�.B%U�B	
�B	�BB&5B�7B��B��B	��B�BVB��B��B�gB$ЌB/n�B�@B�\BXB��B�B!�B�B��B��BsB<fB0VB#mA�KsB�B��BȪB׋B�B�B��B!PlB'jkBƺBԝB�2B{�Bn�B4�BPdB+��B�}BK�A���B ֮B��B�`B��B,wZA�ݟB �'Bx�B!�B�|B%AB	�B	[ B&��B��B�fB�B
>�B��B?B dB�yB�DB%3zB/n�B�{B�~B8�B��B�HB!�B�BRLB��B1�B@~B�B?�A�|tBgB�B��B�BFbB@�B�]B!�B'K	B�QB�MB�B��BCB>�BYB+�6B��BI�A��B ��B+�B��B�B,��A��B>�BL�@!UvA�I@�hIAֻxA���@̜�Al �A�O�Aɫ�A�:�A� �A�ΝA�A��jA�4A;�|Ag��>�3A�ˊAG5�@�l<A�[�A��@��A`��Ah��A��:A��A��@��A��@��RA�O�A���A��kA���A��a@�m�@� w@�AW�6A�W�A���C�C�=���Au�A�T�A%��A�_{@h�A�A6�A�y;A �?�}+@�A�ɆA�B��@�cA��0@��YA�ӯA�H�@���Al'�A���Aɝ�Aɀ&A�~"A��A�p�A���A�IA:��Ag�L>GU�A���AG�@�J~A�]�À@��BA`�+Ahv~AƆ"A�bA���@�M�A�|=@��A�3A�GwA�|�A��A�m�@�[@�@��5AW�A�~xA�v�C�Mz=�G�At�iA���A%��A��@S��A�5A8JA�}}AI?�O�@� 	A�a$A�|AB��            �   
         8      3         	               )   <   
   7                     A      l   %      '   ,      -         !      !      \   &                  *         1   B   
                        /            7            %      +            '   3      !                     )      +         +   #      +                     +                     /         !   #                                       -                  '               +                                          '         #                                          /         !                  N�N&RN]]�O��N�;�N
��Nn7P8
�N7,hO'�IN���O�uNqW�P u�O*VlN�v�N�v�Ot��P:��N�d�O�n�O'�+N	�;M��rOL�hO+�LN7��O}��Og_�O�n�O��rNQ�P �gO���N��O��fNw�N�OR��M��O?UN�l�O���N���N�N�AWN�t�N��OA�P#�OtpNǦ&O��O�+N6��Nђ0N��Ni��NR�  �  j    �  h  �  #  %  w  X  	  K  v  6  (  �  .  }  �  �  X  I  �  m  �  U  r    �  �  Q  m    k  �  �    �  �  M  X  �  
j  F  �  r  ?  �  �    �  |  �  
�  �      I  q����󶼬1=ixռ�t��D���49X;D���ě�<e`B;ě�;�`B;ě�<o<t�<D��<#�
<�1<ě�<D��<���<�C�<�j<�t�<�9X<���<�/=ix�<���=��=\)<�`B=\)=�P<��=,1=C�=0 �=\)=C�=8Q�='�=��=P�`='�=,1=0 �=<j=H�9=T��=Y�=aG�=e`B=���=}�=�o=�7L=�C�=�v���������������������!#-0:50'#"ADLN[gt}���{tg[NLECAZ[gt�������tgb\[ZZZZ,.0<INIG<0,,,,,,,,,,���

�����������������������������%#)6>@<60)%%%%%%%%%%f_[`gqt���������tgff������������������������������������������������������������.-9BZanr}�xqaUH<264.��������������������	

#$-00.$#
		(*68:76*������������������������#/<HS]^[H<#
���@BEGO[]dhhjhb[WOKB@@�������(**(�����������	�����������������������{������������������)57@BCB;5+) ����


���������������).56<<75)TVanz��������zna^VUT��������������������
"/;GIHE?;/"@?ABGOZZ[\[OGB@@@@@@!)5BNckuvtg[N5);88:BN[gtz|��{tg[NB;����������������������).7?CA6)�����6)$)+686666666fghntt���������tshff��������������������RMRUbdebZURRRRRRRRRR*'()-/<HJPQRRPLH@</*,*./<HKUQH><:/,,,,,,)5BIOPNIB85)#%+/<FHJIHE><5/$&%##��������������������xvz��������������������������������������������������
#/<@@B@</,#������)--'�����\UUX]_gmz������zsma\�������������������������������������������������������)'),169;<;:62)))))))��������������������AEHLTTT\YTNHAAAAAAAAz{����������zzzzzzzz�

���������������������������������������������������������������������������������������������һ��������������x�p�l�g�l�x��������������6�B�O�R�X�Y�Q�B�6�)�����������)�6�����������������������������������������'�4�6�:�?�4�(�'�$�%�'�'�'�'�'�'�'�'�'�'�m�y���������y�m�`�Y�`�h�m�m�m�m�m�m�m�m�N�s�����������������g�A���	��*�/�=�NÇÓÖÞÓÇ�z�p�zÁÇÇÇÇÇÇÇÇÇÇ�n�zÇÓØàáäàÝÓÇ�z�n�n�c�c�k�n�nĦĳĶĿ����ĿĳįĦĜěĚĥĦĦĦĦĦĦ�"�;�H�U�a�i�i�a�T�H�;�/�"��� �����"�/�7�<�B�>�<�5�/�*�#�!��#�&�/�/�/�/�/�/���������������������z�a�O�M�R�\�a�m�z��ĦĳĿ������������������ĿĳĦĤĢġĥĦ�4�A�M�Z�f�g�h�f�[�Z�M�A�@�4�(�'�(�(�3�4�T�`�m�n�s�m�l�a�`�T�G�F�A�?�G�L�T�T�T�T�����ùϹܹ���������ܹ�����������������5�L�Z�h�j�e�[�A�(��������������������������������y�s�l�s�|��������'�4�@�M�U�Y�]�_�d�f�Y�M�@�4�$����"�'����&�*�6�=�C�J�C�6�.������������ùü������úùìãëìðùùùùùùùù�����������������������������������������;�G�L�T�T�S�H�G�;�.�"��
��	���"�.�;�T�`�m�����|�y�m�`�T�R�G�@�;�9�;�<�G�R�T�a�n�p�r�n�c�a�a�U�S�U�V�a�a�a�a�a�a�a�a��#�0�<�I�U�\�Z�U�K�I�0�#����	������	���
��������������������������������ûлܻ�����ܻлû����������������;�H�T�a�k�m�u�x�x�m�a�[�;�/�(��"�/�2�;�лܻ����������ܻۻջл̻ллллл��������������������������������������˿ݿ�����(�8�=�3�(������ٿӿϿӿ������������������������������������������/�;�>�>�3�/�"�	�����������������	��"�/�Z�S�Q�Z�g�s�������z�s�i�g�Z�Z�Z�Z�Z�Z�Z�'�4�5�@�M�M�M�M�@�<�4�-�'�����%�'�'���'�.�3�3�'�������������������������������������������������������׾���	���#�#�"��	�����׾оʾž˾�������������������������������������N�[�t�t�g�[�B�3�)�&�%�*�5�ND�EEEEEEED�D�D�D�D�D�D�D�D�D�D�D������ùŹɹù����������������������������������ĿʿĿ���������������������������������������������������ſž�����������ƽ��Ľ˽˽ŽĽ����������������������������5�B�N�[�g�k�o�q�n�g�[�N�B�:�*�)�&�)�+�5������-�:�]�l�l�F�-������Ⱥĺպ���{ŇŔŠŭŹ����������ŹŭŠŔŀ�{�u�t�{����(�4�8�A�L�O�M�A�4�(�$���������������������������������z�n�k�w�|���������ʼּ���������߼ּҼʼ������������e�r�z�~�������~�r�e�Y�W�Y�e�e�e�e�e�e�e�l�x���������������������x�l�h�e�l�l�l�l�
���#�0�0�0�#���
��
�
�
�
�
�
�
�
����������������������������������������ǭǯǭǡǔǋǏǔǞǡǢǭǭǭǭǭǭǭǭǭ d O 3  @ f 6 T ;  2 ' P 8 . * 7 a * < @ H M H   ) A  D * % I 9 1 ? A E & / = G   . L > A @ F  m : u N 5 k S q w ?    E  X  s    >  P  w  v  J  ^  �  �  �  j  q    �  K  -  �  N  {  :    �  p  c  �  �  "    s  _  �  �  �  �  �  �    �  �  �    *  �  �  B  �  �  �    �  ;  |    l  �  b  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  =j  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  X  A  +  j  h  e  b  _  [  U  N  H  A  :  2  +  #         �  �  �  �  �                  �  �  �  �  �  �  �  �  �  �  X  
  �  �  �  �  t    �  �  �  U  �  M  }  w  �  
      h  e  a  \  X  R  K  C  :  0  $      �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  z  m  a  W  M  D  :  0  '  #       �  �  �  �  �  �  �    h  Q  9    �  �  �  �  x  �  �  �  %      �  �  �  �  �  �  �  h  +  �  Q  �  �   |  w  k  `  T  G  :  -         �  �  �  �  �  �  s  I     �  �  �       /  A  N  V  U  L  8    �  �  /  �    T     �  �  �  �    �  �  �  �  �  f  7    �  �  i  +  �  �  Z   �  (  J  K  F  <  +      �  �  �  �  �  �  �  �  6  �  y    ]  h  r  t  q  k  \  N  5    �  �  �  o  :  �  �  n     �    6  0      �  �  �  �  �  �  �  }  v  n  `  F  #  �   �  (         �  �  �  �  Q    �  �  d    �  b  �  c  �    �  �  �  �  �  �  ~  t  j  _  R  A  -    �  �  �  �  n  ^  .  )  $          	     �   �   �   �   �   �   �   �   �   �   �    1  L  h  z  {  k  \  C    �  �  b  �  �    �    �  %  �  �  �  �  �  �  �  �  �  n  &  �  |  3  �  �  ?  �  Y   �  �  �  �  �  �  �  �  �  t  X  :    �  �  �  �  �  q  V  ;  t  �    B  X  N  <  $    �  �  t  +  �  K  �  �  6  �  |  I  B  8  ,        �  �  �  �  �  �  o  W  @  %     �  �  �  �  �  �  �  �  �  �  �  �  �  d  2  �    �  R  �  �    m  f  _  X  Q  J  D  =  6  /  $      �  �  �  }  E    �  �  �  �  �  �  w  g  T  >  $  	  �  �  �  �  c  @    �  %  H  O  S  T  H  8  %    �  �  �  �  �  �  s  S  *    �  �  �      8  S  m  }  �  �  �  �  y  h  G  #  �  �  y  &  �  �  �        �  �  �        �  �  I  �  #  O  {  �  �  �  �  �  �  �  �  �  �  p  �  z  a  G  )    �  �  E  �  ^  
V  
�  
�    X  �  �  �  �  �  l  )  
�  
[  	�  �    �  a  i  �  �    9  O  L  ?  +    �  �  �  �  ^  �  s  �  F  �  �  m  e  ]  T  I  >  3  (        �  �  �  �  �  �  f  �  k  �  �      �  �  �  �  �  �  �  �  �  _  #  �  r    �    2  X  i  j  b  U  B  0      �  �  �  E  �  �    9  4   �  �  �  �  �  �  r  c  S  A  0    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  R    �  M  �  !  �    k  V  B  0      �  �  �  �  z  k  ]  E  !  �  �  �  r    ?  Y  l  y  �  �  �  |  w  i  X  <    �  �  O    H  {  �  �  �  �  �  �  m  O  ,    �  �  h  2    �  �  Q    m  M  G  B  <  6  /  %        �  �  �  �  �  ^  :     �   �  �  "  <  N  X  P  A  '  
  �  �  �  E  �  q  �  <  �  �   �  {  �  �  �  }  n  \  G  -  
  �  �  o  -  �  �  -  �  c     �  	I  	�  	�  
*  
M  
b  
j  
_  
?  

  	�  	�  	&  �  �    �  }  F    !  	  �  �  �  7  
  �  �  E     �  {  3  �  �  (  �  �  �    t  z  q  [  >       �  \    �  �  A  �  �  `     �  r  n  i  c  W  K  =  .      �  �  �  �  �  �  g  4   �   �  ?  *      �  �  �  �  �  �  X  +  �  �  �  I    �  @  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  y  �  �  �  �  �  �  �  �  m  K    �  �  f    �     b  �      l  L  %  �  �  �  |  C  �  �  K  �  {  �  C  �  s  �  7  �  �  �  d  A    �  �  �  _  #  �  �  �  �  �  Z    �  >  |  f  N  5         
  �  �  �  �  d  .  �  �  �  B    �  �  �  h  j  a  Q  7    �  �  �  j  +  �  �    �  �  �  -  	r  	�  	�  	�  	�  
1  
�  
�  
b  
*  	�  	�  	=  �  O  �  w  �  /  ?  �  �  �  �  �  �  �  �  �  �  �  �  �  �  	      *  7  C    �  �  �  �  �  �  �  �  �  �  }  b  3  �  �  ~  :   �   �    �  �  �  �  �  �  �  �  �  �  �  }  q  _  N  0  �  k  	  I  4    	  �  �  �  �  �  �  {  o  t  y  n  O  /    �  �  q  5  �  �  �  z  L    �  �  �  H    �  �  ,  �  d    �