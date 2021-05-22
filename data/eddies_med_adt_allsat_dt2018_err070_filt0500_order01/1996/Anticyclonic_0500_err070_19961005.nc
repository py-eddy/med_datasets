CDF       
      obs    :   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?ɺ^5?|�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P�_      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �D��   max       >�      �  |   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @EǮz�H     	   d   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?ۅ�Q�    max       @vz�G�{     	  )t   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @Q@           t  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Ͻ        max       @��           �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �ě�   max       >��      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�+T   max       B,�/      �  4�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�yQ   max       B,�      �  5�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?9�|   max       C���      �  6�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?N�   max       C���      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  8h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  9P   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5      �  :8   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       PW��      �  ;    speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��ᰉ�(   max       ?��+j��      �  <   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �#�
   max       >�      �  <�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>������   max       @EǮz�H     	  =�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?ۅ�Q�    max       @vz�G�{     	  F�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @Q@           t  O�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ͻ        max       @�9�          �  Pl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B�   max         B�      �  QT   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��t�k   max       ?��+j��        R<            @                        	                              	               �      0   '      <      	   H      5      
   5         %   	               �               y   8      	   O��N ~�N[��P��N�bHN���N��}NGn�N�0O��O�lN���N�fKO���O��kN�LN�uFO��N�^6O���O���N��N�
.Nfg�NPa�O���Pw��O�٩PM�jP�jN��YOƮ�OL�N�bP*��N���P�_O��>O}�O�`O£N��O��4N��sNX��N�>bNk_�O�w O�%�O(�PN��N�ON�P"�$Oj��No��N^�M��D���#�
�t��D��;�o<o<t�<49X<49X<D��<D��<u<�o<�o<���<�1<�1<�9X<ě�<ě�<���<�/<�/<�h<��=C�=\)=\)=t�=�P=��=#�
=#�
=#�
=#�
='�='�='�=0 �=<j=<j=<j=<j=D��=Y�=]/=aG�=ix�=u=�+=�C�=�t�=��=��-=��=�;d=�x�>�)5BNRZZb_NB5) NIEINOP[\[XUQONNNNNN������������������������#089980#
���fb^gt�������tgffffff��������������������xxz|��������|zxxxxxxhhqt������ztphhhhhhh��������������������.+)*/<HLUX`XUH<<</..HOPTX[hstxtonh`[XOH�����������������������������������������������

 ������#0<IU[_^ZUL<0#������		�����������������������������������
#&$ 
����	"/74/-'"!
	��&5<>A85) ������$*,39:6)�������')+)������������ ��������//4<HIKIH?<13/////// #/<>HPU\UHD<//#    �����������������������)5Nm��tgN5)�������
#+/21#
�����qv|���������������tq�������
$/9@@=/#
���)169;6)�����������������������������������������������������������������BNZb]NC5)���������	����������5B[``ZN75)� )1F[ginvsla[NB5)
	
#$/0860.+(#

(5BR[bedOB6)XXX[\bhpt�������th[XW[ahtutsoha[WWWWWWWW������������������YXabmnz}�����znmmaYY��������������������=?BBO[\a`[WODB======�������������������������������������������
 !
��������������$'������������������������#,///1/)#"

#����������������������������������������������

������	
#)$#
ysnnrvz}~���|zyyyyyy���������������������G�T�`�m�t�z�{�}�{�y�m�`�G�;�2�.�2�5�;�G�������ĽнѽнǽĽ����������������������#�/�;�9�5�/�#���#�#�#�#�#�#�#�#�#�#�#����4�K�T�W�A�(����Ͻ������Ľݽ������������������������������������������A�M�R�Z�f�s�t�����s�f�Z�R�M�M�A�?�A�AFFF!F$F0F(F$FFE�E�E�E�FFFFFFF�Y�Y�e�g�l�g�e�Y�W�N�L�K�L�Y�Y�Y�Y�Y�Y�Y�����������û̻û�������������������������������������������������������:�E�F�S�_�b�f�_�]�S�F�B�:�-�%�!��!�+�:����������������������������������������������������������¿²¬±²¿���������˼��'�4�9�?�Q�N�M�@�4�'� �������������������������������r�f�Z�P�P�Y�r�~���[�b�g�t�j�g�[�N�B�5�/�5�=�B�N�U�[�[�[�[�������������ܹ۹۹ܹ����������:�F�J�S�\�^�V�S�L�F�:�-�!��!�"�(�-�/�:�H�M�T�a�i�m�r�s�m�a�V�T�H�<�;�3�;�>�H�H¿������������²¦�}�}�}²»¿�	�.�;�L�N�;�"��	���׾ʾƾþǾȾ̾׾�	�����������������������������������������B�O�V�[�h�i�t�~�t�h�[�O�H�B�=�>�B�B�B�B�Z�^�g�m�i�g�Z�N�L�A�<�A�N�U�Z�Z�Z�Z�Z�Z�"�$�'�#�"����	���	����!�"�"�"�"�������������������������������u�~�������B�S�S�W�]�c�j�[�6�)���������������B������ �(�.�(�����ݿٿѿĿſƿѿ����������	���-�2�/�"������������z�|���.�B�T�`�y�������y�i�T�;�.�������"�.��'�.�3�4�>�3�+�'�����������āčĚĳĿ������������Ŀĳč�r�h�[�a�tā�N�Z�a�c�V�N�B�5�#���
�
����(�6�G�N�O�\�h�s�q�h�b�\�O�C�7�<�C�G�O�O�O�O�O�O�����#�<�J�T�X�Y�\�U�<�#�������������������*�1�0�*�#��������������Ƴ������0�-�����Ƨ�w�C�8�&�'�6�_�uƚƳ��������5�W�^�Z�A�(����������������!�(�:�:�.�-�!��	�����������r�~�����������ɺٺֺɺ������{�g�Y�U�e�r�����������ʼͼʼǼ¼��������������������������������������������������������������ʾ׾���������׾ʾ�����������������ŇŔŞŠŭŭŮŭťŠŔŇŇ��{�{�{ŀŇŇÇÊÓàäàØÓÇ�z�z�x�zÅÇÇÇÇÇÇ�'�4�@�B�K�A�@�4�'������'�'�'�'�'�'�4�A�M�Z�\�Z�M�H�M�N�M�A�<�4�/�/�4�4�4�4�H�U�a�nÇËÍÌÇ�z�n�a�U�U�P�?�:�<�E�HD�D�D�D�D�D�D�D�D�D�D�D�DtDjDjDnD{D�D�D������	��"�+�/�1�0�"��	��������������������������������������������������������ǭǡǔǈǄ�{�s�o�m�o�{�}ǈǏǔǡǭǱǴǭ�������������������������������x�w�y�������������������ܻû�������������EuE�E�E�E�E�E�E�E�E�E�E�EuEnEiEeEcEoEuEu�6�C�D�O�V�O�L�C�6�4�*�)�*�+�6�6�6�6�6�6�ּ�������������ּռּּּּּֽ����ýĽнݽ޽ݽнĽ������������������� # n Y : A � 0 Y V ( A 9 T F + M ' * W > [ I g A � P G a ` J _ < p I 7 > n V I < . < . - ,  J ` + ; h S a 8 : 0 z �  )  ~  }  �  �    �  �  �    *  �  �  ?  N  �  �  A    2    '  �  j  �  �    �    \    �  �  �    �  H  )  &    L  E  j    g  �  �  �  "  �  K  -  �    �  t  �  z<o�ě��o=u<49X<T��<���<�1<�1=\)<�j<�j<�j=\)=8Q�=o=C�<�=\)=0 �=P�`=\)=+=\)=t�=Y�>5?}=q��=�1=���=<j=���=T��=D��=�`B=<j=�v�=q��=Y�=���=�o=H�9=���=e`B=�C�=�o=m�h=�{>��=��
=�t�=� �=��
>J��>!��=��#=��#>��Bd�B��B `xB$~�B	��BDB��B��B#�B��B�B,�B؜B#/XB&B�B mGB$6GA�+TBP�B:�B�BB��B��B��B6�BnGB�B0�B6�B�RB
�B�^B�'By�B��BT B;�B%�B�1B�B��B FHA��IB"]�B�-B��B�uB�B�OBw�B{oB,�/B�B��BD^BC�Bd?B?#B�#B ~�B$OaB	�Bp�B��B��B#E�B�BB��B	B 3�B"�TB&<HB��B H�B$F�A�yQBJ�B'iB�B|?B��B�B��BUHB?�B@)BHpB�B��BBd�B7=BɾBCvBB�B%E{B�]B:�B��B B�A�^[B"@FB��B�?BfB?+B2{BADBG�B,�B�B�pB@�BL/B�wAf�2A$��A�1A2<A�j�A@i�C���?��S@�-�A��}@�@lA�WA���@�²@��aA�̷?9�|@~-$A���A��7AZ$�A��QA�d�A��A�Z�A��]A�!�A�A���Ad�2?���AߵA��B��A�S�A��BxCA��A
rC@*�@��&AI+\ARlwA��A��@̩,A;gPA���C��ZA�}A���B&xAz�@��AC��}B ��A�_A&�-AfZA$� A£ A2
�A�eAAE�C���?ؾ�@���A҃�@���A�x@A�f�@��@��A�uz?N�@��A�|A���AXh(A�/eAڡA�M�A��fA�z�A�r�A�y�A��cAgb?��[A���A���B��A�~(A���B�A���A�@�n@��AH>ZAS hA���Aɩ*@ʃ�A9�A�q|C���A�z#A���BW�A��@���C��B ��A��A(            @                        	                  	            	               �      1   (   	   <      	   I      5         6         &   	               �               z   8      	               )                                                   %               !   7   #   3   )      #         '      =   %      %                           !               '                        !                                                   #               !         1   %               '      5         !                                                      O%X�N ~�N[��O��N�bHN���N��}N&X�N�0N�"�O�lN*A�N�!O���O���N��RN�uFO��N�^6OyX�O���N��N�
.Nfg�NPa�O���O�m�O�}nP>?9O���N��YO��OL�N�A$P*��N���PW��O�EO}�O�-pO£N��O�;�N��sNX��N�>bNk_�O��GO@ӚOs�N��N�ON�On�O%�No��N^�M��  �  �  5  7  �  �  �    �  q  �  �  W  �  �  Y  k  �  &  �  k  b  �  �  ,  �    ]  ;  �  �  N  �    p  [  �  B  �  s  ~     �  j  �  j  �  �  �  �  �  g  �  �  �  �  �  軣�
�#�
�t�<49X;�o<o<t�<D��<49X<���<D��<�t�<�C�<�o<�1<�j<�1<�9X<ě�<���<���<�/<�/<�h<��=C�=��=��=��=#�
=��=,1=#�
='�=#�
='�=D��=49X=0 �=m�h=<j=<j=P�`=D��=Y�=]/=aG�=m�h>n�=�7L=�C�=�t�=��>�=�x�=�;d=�x�>�)*5BINQQUSNB5*) NIEINOP[\[XUQONNNNNN��������������������������#)023430#
��fb^gt�������tgffffff��������������������xxz|��������|zxxxxxxtitt������uttttttttt��������������������///8<@HOUVUMH<4/////HOPTX[hstxtonh`[XOH�����������������������������������������������

 ������#0IUY^\YUJ<0#������������������������������������������
#&$ 
����	"/74/-'"!
	��$)5;=?;65����$),286)��������')+)������������ ��������//4<HIKIH?<13/////// #/<>HPU\UHD<//#    �������������������� ���)5BEEB;5)	 �����
#'/1/#
�����vw}����������������v��������
#/8??;/#
�)169;6)�����������������������������������������������������������������BNZb]NC5)���������	��������)5NZ]]WN5)').5BJS[efjomgb[NB5'
	
#$/0860.+(#

!/6BOX]ZZOB6)XXX[\bhpt�������th[XW[ahtutsoha[WWWWWWWW��������������������YXabmnz}�����znmmaYY��������������������=?BBO[\a`[WODB======������������������������������  ���������������

��������� #%�������������������������#,///1/)#"

#����������������������������������������������

�������	
#)$#
ysnnrvz}~���|zyyyyyy���������������������T�`�c�l�m�q�n�m�j�`�T�G�;�8�6�7�;�;�G�T�������ĽнѽнǽĽ����������������������#�/�;�9�5�/�#���#�#�#�#�#�#�#�#�#�#�#�������(�@�E�A�6�(������۽ʽ̽ݽ������������������������������������������A�M�R�Z�f�s�t�����s�f�Z�R�M�M�A�?�A�AFFF!F$F0F(F$FFE�E�E�E�FFFFFFF�L�Y�e�e�k�e�]�Y�Y�O�L�L�L�L�L�L�L�L�L�L�����������û̻û���������������������������������������������������������:�E�F�S�_�b�f�_�]�S�F�B�:�-�%�!��!�+�:��������������������������������������������������������¿²®²³¿�����������ؼ��'�4�9�?�Q�N�M�@�4�'� ����������Y�f�r���������������������u�f�\�R�S�Y�[�^�g�k�g�e�[�N�B�6�B�B�N�X�[�[�[�[�[�[�������������ܹ۹۹ܹ����������:�F�J�S�\�^�V�S�L�F�:�-�!��!�"�(�-�/�:�H�M�T�a�i�m�r�s�m�a�V�T�H�<�;�3�;�>�H�H¿������������²¦��¦³¿�.�;�J�K�=�.�"��	���׾ʾžȾɾξ׾�	�.�����������������������������������������B�O�V�[�h�i�t�~�t�h�[�O�H�B�=�>�B�B�B�B�Z�^�g�m�i�g�Z�N�L�A�<�A�N�U�Z�Z�Z�Z�Z�Z�"�$�'�#�"����	���	����!�"�"�"�"�������������������������������u�~���������)�6�;�B�I�J�K�D�6�)������������ݿ�������&�+�(�������޿ѿͿɿѿ��������������+�0�/�"�����������������"�.�;�T�`�y�������y�g�T�;�.����� �	�"��'�.�3�4�>�3�+�'�����������āčĚĳĿ����������Ŀĳč�u�h�]�c�m�tā�N�Z�a�c�V�N�B�5�#���
�
����(�6�G�N�O�\�h�q�o�h�`�\�O�J�C�?�C�I�O�O�O�O�O�O�����#�<�J�T�X�Y�\�U�<�#�������������������*�1�0�*�#����������������������� �'������Ƨ��M�G�V�uƎƳ�̿�����+�5�A�P�N�A�5�(����������������!�(�:�:�.�-�!��	�����������~�����������ú̺ɺ����������p�b�a�k�r�~�����������ʼͼʼǼ¼����������������������������������������������������������������ʾ׾���������׾ʾ�������������ŇŔŞŠŭŭŮŭťŠŔŇŇ��{�{�{ŀŇŇÇÊÓàäàØÓÇ�z�z�x�zÅÇÇÇÇÇÇ�'�4�@�B�K�A�@�4�'������'�'�'�'�'�'�4�A�M�Z�\�Z�M�H�M�N�M�A�<�4�/�/�4�4�4�4�H�U�a�n�zÇÊÌÌÇ�z�n�a�U�U�Q�G�@�;�HD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DD�D�D�D����"�)�/�0�/�/�"��	�������������	������������������������������������������ǭǡǔǈǄ�{�s�o�m�o�{�}ǈǏǔǡǭǱǴǭ�������������������������������x�w�y����ûлܻ��������ܻлû�������������E�E�E�E�E�E�E�E�E�E�E�EuEsEmEjElEuE�E�E��6�C�D�O�V�O�L�C�6�4�*�)�*�+�6�6�6�6�6�6�ּ�������������ּռּּּּּֽ����ýĽнݽ޽ݽнĽ������������������� ) n Y + A � 0 U V % A D X F - G ' * W 9 ] I g A � P  a ^ H _ : p 5 7 > c O I : . < & - ,  J \   % h S a   0 z �  g  ~  }  �  �    �  K  �  �  *  O  �  ?  "  �  �  A    �  �  '  �  j  �  �    c  �  /    �  �  �    �  '  J  &  f  L  E      g  �  �  9  �  E  K  -  �  �  W  t  �  z  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  �  �  �  �  �  �  �  �  �  �  �  }  U  (  �  �  }    �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  5  "    �  �  �  �  �  �  �  q  a  T  H  =  5  -  *  )  '  v  �    0  7  2      �  �  �  n  =    �  D  �    m  e  �  �  �  �  �  �  v  d  R  A  /      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  M    �  �  L  �  �  =  �  l     �  	              �  �  �  �  �  �  �  �  i  8    �  �  �  �  �  �  �  �  �  �  x  n  f  _  W  N  >  ,      �  �  *  =  M  [  f  p  p  g  Z  G  0    �  �  �  D  �  �  Y  <  �  �  }  q  b  Q  =  '    �  �  �  �  �  �  �  �  F  K  *  �  �  �  �  �  �  �  �  �  �  �  �  �  b  ?    �  �  �    U  U  V  Q  @  .  $          $  8  K  o  �  �    Y  �  �  �  �  �  �  �  {  b  @    �  �  �  �  �  m  I  4  =  J  �  �  �  �  �  �  �  �  \  6    �  �  �  �  �  h  F  I  p  1  ?  K  U  V  Q  D  4  #      �  �  �  �  �  �  �  �  X  k  b  Z  T  O  Q  T  R  G  7  !    �  �  �  Q    �  �  O  �  �  �  �  �  �  �  ~  r  e  W  G  8  ,           	     &      	  �  �  �  �  �  �  q  H    �  �  �  F    �  j  �  �  �  �  �  �  �  �  �  q  \  E  -    �  �  �  S  �  \  g  i  c  \  S  F  3      �  �  �  s  F    �  �     �   �  b  \  U  L  A  4  $    �  �  �  �  �  ~  f  K  0    	  �  �  �  �  �  �  �  �  z  h  U  B  /      �  �  �  �  �  v  �  �  �  �  �  �  �  �  �  b  A    �  �  �  ~  X  >  $    ,  +  )  (  %        �  �  �  �  �  �  �  �  o  ]  J  8  �  �  �  �  �  �  r  `  E  "    �  �  �  �  v  Y  1  (    
w  
�  �    �  W  �  �  �       �  �  H  �  �  
�  	=  W  �  0  U  \  [  R  <    �  �  �  r  E    �  �  �  �    �  $  1  8      �  �  �  �  �  �  �  p  1  �  �  I  �  �    �  �  �  �  �  �  �  �  v  L    �  �  �  �  �  {    �  :  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  W  @  *    �  s  /  M  A  )    
�  
�  
g  
"  	�  	�  	2  �  ^  �  G  �  $  �  �  �  �  }  j  U  @  )    �  �  �  �  }  W  4    �  �  u  B  �  �         �  �  �  �  �  �  �  �  m  U  5    �  �  z  p  G    �  �  �  4  R  ^  a  W  7    �  v  �  3  l  �  �  [  T  L  E  A  M  Y  e  h  ^  T  J  :  %    �  �  �  �  �  v  �  �  �  �  �  �  Y  9    �  �  v  +  �  >  �  �  6   �  /  8  >  B  ?  9  /  "      �  �  �  �  �  �  l  *  �    �  �  �  �  �  �    �  �  �  �  y  n  a  T  F  *    �  �  �    <  ^  r  o  [  :    �  �  h    �  =  �  �  �  e  �  ~  z  v  l  b  c  [  B  !  �  �  �  �  [  &  �  �    T  �     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  `  }  �  �  �  v  `  F  )    �  �  �  a  0  �  �       �  j  [  L  :  &    �  �  �  �  �  m  M  -    �  �  �  �  �  �  �  �  o  S  3    �  �  �  U    �  �  r  7  �  �  l  �  j  ]  P  @  0      �  �  �  �  �  �  v  /  �  z    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  l  �  �  �  �  �  n  F    �  �  G  �  �  �  K  �  R    �  H  �  �  U  �  :  �  �  �  �  �  �  X  �  �  �  r  �  {  �  x  �  �  �  �  �  |  h  S  9    �  �  �  �    ^  8      �  �  �  �  �  �  �  �  �  �  �  {  U  /  	  �  �  �  �  n  Q  g  d  a  ]  V  O  E  7  $  
  �  �  �  y  I    �  z  2  �  �  �  �  �  �  �  �  v  a  H  /    �  �  �  �  �  �  �  r  	�  
�  
�  V  �    Z  �  �  �  ~  T  	  �  
�  	�  	  �  �  �    q  �  �  �  �  �  �  r  C  
  
�  
S  	�  	  S  {  �  �  /  �  �  �  �  a  1  �  �  �  q  6     �  �  T    �  �  4  �  �  �  s  e  W  J  ;  ,        �  �  �  �  �  �  �  |  u  �  �        1  c  �  �  �                    