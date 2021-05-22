CDF       
      obs    @   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�l�C��        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�E   max       P��*        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �� �   max       <ě�        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�����   max       @F9�����     
    �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @v�
=p��     
   *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @2         max       @Q`           �  4�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�t�       max       @���            5,   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �"��   max       ;ě�        6,   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B/~o        7,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��    max       B/�.        8,   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >Q�   max       C��}        9,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       <��   max       C���        :,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �        ;,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          K        <,   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3        =,   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�E   max       P?�q        >,   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�`�d��8   max       ?ӓݗ�+k        ?,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �� �   max       <�C�        @,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�����   max       @F#�
=p�     
   A,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�     max       @v�
=p��     
   K,   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @O@           �  U,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�t�       max       @��@            U�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >   max         >        V�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�*�0�   max       ?ӓݗ�+k        W�   �                                    .                     
            6   
          '                                    A                        9      	                        -                  P��*O zKN�>�NƷN��N��NP�N���N�]OC�KN���O��LO�j�OM�WOoǨN'�OA�N�C:OM�N�hN"�O�kOWf>O΋�Nrx\OC�O��hO��O:3�OL��N|4�O���N���O�T0N��iO4 O���NݢN���P��7N���Ok�,OI�N���N)=�N�>NY��PC��O��N�J�OFhvN)��O)A(N�M�N���O	�WN��9O�_SN�EO!��NdHNhG�N��VO�	<ě�<���<u;�`B;��
�ě��ě���`B�o�t��#�
�D���T���e`B���㼛�㼣�
���
���
��1��9X��9X��9X��j��j���ͼ�����/��`B��h��h�����o�o�o�o�+�+�+�C��C��C��\)�\)�\)�t���P��P�#�
�,1�49X�<j�<j�D���D���P�`�T���Y��u��\)������ Ž� �����)065������|}�#/3<@HNHF</#)130*)	!"#'/;AFFA;/$"!!!!!!����������������������������������������'6BO[_ffposoOB4# !'��������������������./19<HUXaca`^VUH</..�����������������������������������������������������������*6CKZ]TIC*�����#/<HUnva`c\YU</(# )/5BNSRSVUNKB5)! ")X[hrt���th[YXXXXXXXX#/<HU^UPA</(#pty��������tplppppppgn{�����������{okfdg��&)*)�����
 







�����
	����������#0?GIRTRF><0%$
nbI<0*()17DIQblkqqun#)/6;/#"��������������������Lat�����������tgNHGL������������������Ibn{~�������{xuib]UI�����
(20&#
�����Z[httz{th[YOZZZZZZZZ6AIUn{�wxvnmbUI<5516)6BFIEB;6)#�����������������������������������LN[gst�����tg[YNBFLL��������������}|~��}����������������}}�������

���������{�����&%������{{�������������������������������������������������������������������������������������������������������������gmz���{zumfbggggggggx�����������������vx�����������������}}�]anz������zqnka_]]]]u�������������zvutu}����������}}}}}}}}}[bgt�����{xvtg][WUV[�������������������������������������������	
���������),43+)#(/8<?<HNRMF</#tz��������{zuttttttstz�������������xqqs��������������������KOP[dhlih[OMKKKKKKKK����
	������������	����������ػλջ�'�@�Y�t�������v�Q�E�=�4������������������������������������������Һֺ˺˺Ѻֺ�������������ֺֺֺֺֺ�����������������������	���������������������������*�+�6�C�6�*�$����(�%�!�(�4�A�M�X�Z�b�Z�M�A�4�(�(�(�(�(�(�"���������"�G�`�m�y�������m�T�;�.�"�B�<�6�)��#�)�6�B�O�T�O�N�O�Q�O�B�B�B�B�H�C�;�/�*�%�#�"�"�"�&�/�;�H�I�O�M�I�H�HàØÓËÇ�|�z�v�zÓàð��������üùìà���������������������������������������������������v�m�r��������������þ¾ľ����.���������"�;�G�J�Q�T�^�W�T�G�;�.�H�>�<�B�L�O�P�U�^�a�n�zÇÊ�}�q�a�]�U�H��������������*�<�C�L�L�L�C�6�*���e�[�]�e�i�r�r�r�w�x�u�r�e�e�e�e�e�e�e�e�6�-�)�$�$�&�&�$�)�2�B�[�a�c�[�X�U�O�B�6�x�r�l�f�f�l�x���������������x�x�x�x�x�x���������������Ľнݽ�����нĽ������$��!�$�%�0�3�=�F�I�J�L�J�I�B�=�4�0�$�$�ݿۿݿ������������ݿݿݿݿݿݿݿݽ��������������Ľнݽ���ݽнĽ��������������������������ĽннƽĽ������������M�X�q�v�r�f�Z�(��������������(�A�M¿´³¿����������������¿¿¿¿¿¿¿¿�������۾׾����	��"�'�.�/�.� ��	���	��������������� �	��+�#����)�"��	���Y�M�@�2�1�@�M�Y�`��������ļü��������-�'�%�*�-�<�F�S�_�l�{�����x�l�F�>�:�2�-�$�"����
���$�0�=�I�L�O�Q�K�I�=�0�$��������������������������������������лû��������ûܻ�����#� �����ܻ��#�������#�/�3�<�?�>�<�2�/�#�#�#�#�<�0�#���#�0�9�I�b�n�{łł�~�{�o�U�I�<Ň�~�{�z�{łŇŔŠťťŠśŔŇŇŇŇŇŇ������������	��������	�����¿����������Ŀѿ������������ѿ�ŭŦŠŔŎŏŔşŞŠŧŭŹŻſžŻŹŭŭ��������������������������������������'���1�M�����ʽ��.�5����㼤���r�@�'�`�T�T�I�G�;�6�;�G�T�`�a�m�r�r�m�`�`�`�`����׾;þȾ׾����	���"�.�2�.�"����b�V�I�A�@�=�0�0�=�A�V�b�o�{ǅǊǁ�{�o�b�����������������$�$�$�$�������������������������������������������������H�D�;�/�"��"�/�;�H�M�H�H�H�H�H�H�H�H�HƳƯƬƫƳƹ������������ƳƳƳƳƳƳƳƳ�����p�c�^�_�f�t�������ɺ��!���ֺ����S�G�F�:�8�/�:�C�F�S�\�_�l�x�{�x�v�l�_�S�������������������	������������������������������	��"�#�'�$�#���	����������������
���
�������������������ęčĈāąčęĚĦĳĿ����������ĿĳĦę�
��
��
��
����#�)�&�#�#��
�
�
�
�������(�1�5�9�A�F�A�8�5�(�'����g�Z�\�_�g�o�s���������������������s�g�g�n�m�n�x�zÇÓàäçàÓÇ�z�n�n�n�n�n�nE�E�E�E�E�E�E�E�E�E�E�F$F=FEF9F(F
E�E�E�FVFOFJFEF=F=F=FJFKFVF[FcFcFcFVFVFVFVFVFV���
��������#�/�<�H�M�N�H�B�<�/�#��ֺԺκɺźƺɺԺֺ�����ֺֺֺֺֺֹù����������ùϹѹڹҹϹùùùùùùùý�������!�.�8�3�.�"�!�������V�S�G�?�G�S�Y�`�l�y���������������y�l�V 2 B * - H Q 5 B M D 0 F U Z I n B > ? t ` $ 8 3 m 7 d \ } 3 A + ' 5 ' V D > n � i b . c 9 r d ? 0 f * q T J \ ' Q ^ z A e & 5 [  r  7    �  �  �  n  �    �  �  L  �  �  �  �  �  �  �  &  k    �  �  �  �  �  W    �  �  i  �  }  �  n  ;      �  �  =  �  �  Q  ]  n  k  C     �  �  ~  �  �  3  �  Q  n  u  �  m  �  y�"��;ě���`B���
%   �D�����e`B��o��`B�u���m�h���\)�����#�
�C��,1�����ͽ+�'��-�+���ixս�����D���+�,1�'q�����0 ŽP�`���8Q�ȴ9��w�e`B�L�ͽ'�w����w��vɽy�#�D���q���<j�m�h�H�9�q����7L�aG��ě��e`B���㽗�P��^5�Ƨ�ȴ9B|�B�B� A���BR�B�	Bk�B�"B7�BSHB|xB �5B/~oB��B��B�LBR�BωB(�B��B۴B#S�B%��B&��B��B��B
�!B u�B(P�B� B�.B';�B��BT�B��B	F�B)��BB#)IB-�BBjB�Bf�B�\B7HB�
A�/�B=�B^:B�B ��B
�oB	�yB�
B�BB�Bh/BwB�zB
��B"{�B<�Bk�B��B@B=�B��A�� BL�B6�BD�B"xB�BΤB��B!A?B/�.BH�BERB�BS�B�1B)=OB�gB��B#E�B%@�B&��BB�B��B
��B -�B)PB:�B��B'%�BqOBC�B��B	�lB)��BpB#AB-@�B58B@JB@AB��BA�B�dA��B�BI?B�B ��B
�B	@aB<B:B��Bo@B?�B��B
�B"��B@BBVB�(@���A��z@DʖA� A�v�A:�AfMcA�#8A���A���A�FAI[hA_�2A�OA��6?���A�@F@��gA%V�B
��A�A'�A"?A9VAA�<AY��A���@��O@��B
95A�� @�EqA���A��A�5�AZ9wA}u�A�Q�A��!@�?YAg?=AW�=BB՗A�lA��B�@)@]@��pA���A���A�GA��:A�hA�;A��Aɩ>C��}C��}A���@<�p>Q�A$�A!@� �A�v�@E9�A�A��/A;!Ac�A�jxA���Â�A��AAH��A^�pA�~A���?���A���@�r�A%4�B
��AK�A'�A ��A8�ZA��AY�7A�Z�@��@�:B
�sA�`;@���A�{?A���A�jA[ \AɂA�;�A��4AD�Ah�NAV�B�bB	=�A��A��B��@�K@�� A��A���A�m=A�MA��A��@A�}�Aʋ�C���<��A�m$@82>Gn�A�A��   �                        	            .                                 7             '                                    B                        9      	                        -                     ?                  '                  '                                 "         +   +                                    K                        3                              '                                       %                                                                                                   3                        1                              #                  O�GN�<�N��8NƷN��N��NO�pN|�)Nr��O$��N���O��LO|cN��EOoǨN'�O=N�C:O1��N0�N"�O�kO%�(O;�QNrx\O3��ObkOb*O:3�O9�N|4�O6N���O��qN��iN��	O��N�*�N���P$q�N���O&"OI�N���N)=�N�>NY��P?�qO��N�J�OndN)��O)A(N�M�N���O	�WN��9O���N�EO!��NdHN?q�N��VO�	  �  �  �    �  p    K  �  1  s  B  ^  �  �  �    7  �  �  �  �  �  �  �  �  �  �  �  �  �  q  .    "  4      �  A  �    '  @  !  p  �  =  �  �  r  �  ;  �  !  �  O  
Y  �  �  �  S  �  ����<�C�<D��;�`B;��
�ě��#�
�o�#�
�49X�#�
�D�������
���㼛��ě����
��9X�ě���9X��9X���ͽ49X��j�����#�
�,1��`B����h�o���t��o�\)�+�C��+�D���C���w�C��\)�\)�\)�t�����P�#�
�8Q�49X�<j�<j�D���D���P�`�Y��Y��u��\)���-�� Ž� ������	����������#/1;<H></#).1-)&
!"#'/;AFFA;/$"!!!!!!����������������������������������������&,6BO[^clkhldOB9(%#&��������������������<<HUX[UQH<54<<<<<<<<���������������������������������������������������������� *6=CFLHC6*�� "#/8<HMUWYUHD<2/##"")/5BNSRSVUNKB5)! ")X[hrt���th[YXXXXXXXX#/<AHOLHE><//#pty��������tplppppppjn{����������{qnigfj�!��������
 







�����
	����������#08<EIHOB<0'#/026<IU^aa_ZUNI<10//#)/6;/#"��������������������st�������������tknss��������������������Ibn{~�������{xuib]UI����
#'/1/%#
�����Z[httz{th[YOZZZZZZZZ;EIUbjssrnkfbUI=975;)6BFIEB;6)#�����������������������������������NN[gt~����tga[NNNNNN���������������~}~������������������������

�������������
����������������������������������������������������������������������������������������������������������������������gmz���{zumfbggggggggx�����������������wx�����������������}}�]anz������zqnka_]]]]xzz�������������{zyx}����������}}}}}}}}}[bgt�����{xvtg][WUV[�������������������������������������������	
���������),43+)#)/<?HMQPLE</+#tz��������{zuttttttstz�������������xqqs��������������������LOS[bhkhh[ONLLLLLLLL����
	������������	����������������'�4�@�M�Y�^�b�_�U�@�4�'�����������������������������������������ҺֺϺͺӺֺ������ �������ֺֺֺֺֺ�����������������������	���������������������������*�+�6�C�6�*�$����(�%�!�(�4�A�M�X�Z�b�Z�M�A�4�(�(�(�(�(�(�.�"�����"�.�G�T�a�m�y�������m�G�;�.�B�>�6�)�%�)�+�6�B�O�R�O�L�N�B�B�B�B�B�B�/�/�(�,�/�;�E�H�L�J�H�;�/�/�/�/�/�/�/�/àÚÓÒÎÇÄÀÇÓàìù������úùìà���������������������������������������������������v�m�r��������������þ¾ľ����"��	���������	�"�.�:�A�E�I�H�G�;�.�"�H�F�C�H�J�S�U�\�a�n�v�s�n�g�a�`�U�U�H�H��������������*�<�C�L�L�L�C�6�*���e�[�]�e�i�r�r�r�w�x�u�r�e�e�e�e�e�e�e�e�5�)�'�&�)�)�+�6�B�I�O�[�^�\�[�T�O�I�B�5�x�r�l�f�f�l�x���������������x�x�x�x�x�x�����������������Ľнݽ��ݽнĽ��������0�-�*�0�7�=�I�I�I�G�?�=�0�0�0�0�0�0�0�0�ݿۿݿ������������ݿݿݿݿݿݿݿݽ��������������Ľнݽ���ݽнĽ����������������������������Ľ̽ý��������������A�4�(���	���(�4�A�M�T�Z�^�d�Z�T�M�A¿´³¿����������������¿¿¿¿¿¿¿¿�������ݾپ�����	���"�%�.���	�����������������������	�	��	�����������r�f�Y�T�M�H�A�M�Y�f�r���������������-�'�%�*�-�<�F�S�_�l�{�����x�l�F�>�:�2�-�0�&����� �$�.�0�=�I�K�O�P�J�I�=�7�0��������������������������������������лû������ûܻ��� ���������ܻ��#�������#�/�3�<�?�>�<�2�/�#�#�#�#�<�0�#���(�0�<�b�n�t�{�~��y�n�e�U�I�<Ň�~�{�z�{łŇŔŠťťŠśŔŇŇŇŇŇŇ�������������	�������	����������Ŀ��������Ŀѿ��������������ѿ�ŭūŠŕŔőŔŠŢŭŸŹŽŻŹŶŭŭŭŭ��������������������������������������f�[�������ʼ���#�'�#������㼤���f�`�T�T�I�G�;�6�;�G�T�`�a�m�r�r�m�`�`�`�`���	�����׾ӾʾʾӾ׾���������b�V�I�A�@�=�0�0�=�A�V�b�o�{ǅǊǁ�{�o�b�����������������$�$�$�$�������������������������������������������������H�D�;�/�"��"�/�;�H�M�H�H�H�H�H�H�H�H�HƳƯƬƫƳƹ������������ƳƳƳƳƳƳƳƳ�����r�e�_�`�f�u�������ɺ�����ֺ����S�G�F�:�8�/�:�C�F�S�\�_�l�x�{�x�v�l�_�S�������������������	����������������������������������	�
�� � ����	��������������
���
�������������������ęčĈāąčęĚĦĳĿ����������ĿĳĦę�
��
��
��
����#�)�&�#�#��
�
�
�
�������(�1�5�9�A�F�A�8�5�(�'����g�Z�\�_�g�o�s���������������������s�g�g�n�m�n�x�zÇÓàäçàÓÇ�z�n�n�n�n�n�nE�E�E�E�E�E�E�E�E�FF$F=FDF8F1F'F	E�E�E�FVFOFJFEF=F=F=FJFKFVF[FcFcFcFVFVFVFVFVFV���
��������#�/�<�H�M�N�H�B�<�/�#��ֺԺκɺźƺɺԺֺ�����ֺֺֺֺֺֹù����������ùϹϹعйϹùùùùùùùý�������!�.�8�3�.�"�!�������V�S�G�?�G�S�Y�`�l�y���������������y�l�V  @ ! - H Q 1 9 @ D 0 F C { I n / > ? g ` $ 8 3 m 8 & Z } + A  ' 5 ' T C 5 n i i P . c 9 r d = 0 f 3 q T J \ ' Q N z A e # 5 [    �  �  �  �  �  �  �  r  s  �  L      �  �  9  �  �  �  k    g  �  �  t  ,  �    �  �  �  �  1  �      �    j  �  �  �  �  Q  ]  n  X  C       �  ~  �  �  3  �  �  n  u  �  Q  �  y  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  �    	  	�  
  
c  
�     k  �  �  �  �  �    
f  	�  0    {  �  �  �  �  �  �  �  �  �  �  t  U  3    �  �  �  `  )  �  �  �  �  �  �  �  �  �  w  Z  6    �  u    �  I  �  b  �    �  �  �  �  �  �  �  �  �  �  x  i  Z  L  ?  -    �  �  �  �  �  �  �  �  �  �  �  �  �  ~  u  n  f  _  @     �   �  p  o  m  l  i  e  b  a  a  a  c  e  g  k  p  t  v  j  ]  P  �           �  �  �  �  }  l  j  g  [  ?    �  �  �  p  I  J  K  I  D  >  7  1  +        �  �  �  �  }  Y  1  
  �  �  �  �  �  �  �  �  �  �  �  {  k  T  =    �  �  �  E  !  0  1  /  '        �  �  �  �  `  .  �  �  �  1  �  1  s  n  i  c  ^  X  Q  K  B  8  -  #      �  �  �  �  �  �  B  5  #        �  �  �  �  �  �  �  ~  ^  8    �  �  S  �  �  �    +  D  W  ^  W  @    �  �  �  .  �  h  �  S  j  ^  R  L  J  l  �  �  �  �  �  �  q  `  �  _    �  {    �  �  �  �  �  �  �  �  p  [  L  ;    �  �  �  `  =    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  y  �  �  �  �  �           �  �  �  �  Z  $  �  �  �  U  7    �  7  (      �  �  �  �  �  �  ^  6  
  �  �  �  s  c  c  }  �  �  �  �  �  |  \  9    �  �  �  |  E     �  E  �  Q   �  �  �  �  �  �  �  �  �  o  U  4    �  �  �  N    �  �  y  �  �  �  �  �  �  �  �  �  �  �  x  j  X  <        �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  ^  D    �  �  j  t  �  �  �  �  �  �  �  m  G  !    �  �  �  �  i  2  �  �  \  �  �  8  ~  �  �  �  �  �  �  Z    �  y    x  �  �  	  �  �  x  c  O  :  %      �  �  �  �  �  �  p  )  �  }  "  �  �  �  �  �  |  p  `  M  7    �  �  �  d  1    �  �  �  �    d  }  �  �  �  �  �  �  �  �  �  �  z  ?  �  �        C  x  �  �  �  �  �  �  �  �  �  [    �  v    �  )  �  �  �  �  {  l  R  9      �  �  �  �  a  -  �  �  �  �  �  �  �  �  �  �  �  �  y  Z  9    �  �  �  V    �  �  ;  G  �  �  �  �  �    w  o  g  ^  U  I  >  2  '       �   �   �  b  ^  a  n  m  g  _  S  D  /    �  �  �  �  �  h  E     �  .  "              	  �  �  �  �  �  p  L  #  �  �  ~  �            �  �  �  �  X  .    �  �  �  V  �  2  �  "        �  �  �  �  �  �  �  �  �  ~  q  c  V  K  @  5      !  ,  2  3  -  $      �  �  �  �  [  0  �  �  y  2        	  �  �  �  �  �  i  @    �  �  f  $  �  �  [   �                �  �  �  �  �  �  �  �  �  �  �  p  \  �  �  u  `  K  6       �  �  �  �  Q    �  �  6  �  �  Q  �  �  �  �  3  1  *  ,    �  e  �  �  +  �    |  �     �  �  �  �    w  r  m  g  W  7    �  �  �  �  w  V  4     �  �  �  �  �      �  �  �  �  �  \    �  k    �    �   �  '      �  �  �  �  �  ^  5    �  �  y  =  �  �  j  "   �  @  %  
  �  �  �  �  �  �  �  }  o  a  X  Z  ]  b  t  �  �  !  $  &  )  ,  /  2  5  9  <  D  Q  ]  j  v  �  �  �  �  �  p  i  a  Z  S  K  D  9  ,         �  �  �  �  �  |  a  F  �  �  �  �  �  �  �  �  ~  {  x  u  r  h  O  7       �   �  6  7     �  �  �  T    �  �  >  �  y  �  �  O  �  g  �  s  �  �  �  �  p  [  G  2      �  �  �  y  J    �  �  '  �  �  f  H  ,    �  �  �  �  �  w  `  K  =  0  #      �  �  f  i  l  p  m  c  U  C  /      �  �    	    -       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �        ;  0  %        �  �  �  �  �  p  �  �  ^  '  �  �  b    �  �  �  �    {  w  n  c  W  L  A  6  (       �   �   �   �  !  
  �  �  �  �  �  �  i  O  4    �  �  �  �  ~  Y  1    �  �  �  �  i  F    �  �  �  Y  &  �  �  �  �  1  &    
  O  I  D  >  8  2  +  $          �  �  �  �  �        	�  
L  
-  	�  	�  	�  	;  �  }  �  b  �    o  �  *  �    '  $  �  �  �  �  �  �  �  �  �  �  {  k  \  N  C  9  .  $      �  �  �  �  �  �  �  |  Z  0  �  �  �  N    �  �    c   g  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  
    '  5  @  K  Q  U  W  T  H  4    �  �  �  c  /  �  �  `  �  �    �  o  ^  K  7  "    �  �  �  �  n  N  2      +  a  �  �  �  �  �  �  �  �  �  {  b  H  .    �  �  �  M  �  �  j  A