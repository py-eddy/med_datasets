CDF       
      obs    E   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�KƧ       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�w�   max       P�9�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =D��       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?#�
=p�   max       @F�Q��     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @v��
=p�     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P�           �  6x   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�T�           7   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �C�   max       ='�       8   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�ܚ   max       B4�       9,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�A�   max       B4�A       :@   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��2   max       C�Ѵ       ;T   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >���   max       C��L       <h   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          [       =|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          M       >�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          E       ?�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�w�   max       P��       @�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���3��   max       ?�1&�x�       A�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       =D��       B�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?#�
=p�   max       @F�Q��     
�  C�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @v��
=p�     
�  N�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P�           �  Y�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�%�           Z   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�       [$   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��$�/   max       ?�1&�x�     �  \8            9   	                                 <   #   
               [      #                                  '   (            1   +      /      %      	         	   $   
                      9      	      %         	      "NX��O1(NxxO�D�N�+NQ{nO���O��O�X(O��{M�w�N�$�Nb:O��ZO�nP���P4�"N�4OAO� �NRN�N1��P�9�N@9�P8�jN�'N���P/�N�{LO�[NWN���Nϼ!N��N��O�6wOud�O�TNP-CO��O�<{O�K�O�>�O��ON��O�K�N�հN�T�N��kNv�N!0'O�_O/��N�/gO�Oe��N9�OZ�O��O��Oy�N��lO�DO�JrN���N��N�2FO:�&O�=D��=t�<�t�<�o<t�<t�:�o:�o:�o�D�����
���
��`B�o�t��t��t��#�
�#�
�49X��o��o��C���C����
��1��1��9X�ě��ě��ě�����������h��h�+�C��t��t���P��w�#�
�'',1�,1�8Q�<j�<j�<j�@��@��@��D���D���L�ͽL�ͽL�ͽP�`�y�#�����+��\)��\)������
��Q콸Q������������������������������	
��������������������������������������������<BDN[`ggjig[NB<<<<<<������������������������
/88#
��������#)-/9;BCC<3/*#"CHUanz�������znaUH=C���
#8BE:#
��������vz������zzvvvvvvvvvv&)+6>BJOOOOGB<63,)&&fhltw�����tlkhffffff
#/?HMPMFB</#���
.19=QTZad__XQH;7/*-.���#0b{������b<0���������������������������������������������������������������)BNt���������tN@):BOU[^[UOJB;::::::::pt�����tikppppppppppluz������������shbhl),+)&�����
������yz������������zyyyyy�������������������������������������||���������������������MPYft���������th[TMM������������������������ 


������������

����������������������������������������������
�������z�����������������{z'/<HMU[bfhdaURI</(%'%*/6CCIKC6*)%%%%%%%%��������������������Z_cht�����������g[VZ����������������~zy������ ��������������������������25BMNSSNLB?775222222ACOht}����th[QPLJCA��������		����������ntz������������tnnnn���������������������������������������������������#1HU\bgjcaU</#���  " ����(/8<?Uaja\[VUHF</.+(LO[hssptzythg[TOKMJLU[gt����������tgZOOU��������������������agnpz���������zofaa[dnz�����������naZV[#/<CGFD?/#-15;=:;?B5* ��������������������?BCNN[\`ec][UNBB????Z^]bgt�������tg[TUYZ��������������������#/0670#"qt����������utonqqqq��������������������6<<HKUaakha^USH<9466E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��¦²»ºº´²¦¤��������������������������������������/� ��� �+�<�H�U�a�n�s�x�|�|�q�a�U�H�/�������������������������������������������������������ɾľʾ˾ʾ����������������Z�N�A�>�?�K�N�k�s�������������������s�Z�<�5�/�#��#�/�<�H�U�a�n�w�zÂ�z�n�U�H�<�g�d�U�H�A�L�Z�g�����������������������gùìØ×ÓÔàìù��������������������ù���������������������������������������Ҽr�j�f�]�f�i�r�|�����������������y�r�r�H�C�<�7�<�H�I�U�a�c�f�a�U�L�H�H�H�H�H�H�!�%�&�)�0�6�B�O�[�h�q�s�l�r�h�[�O�B�6�!�������s�Z�T�Z�f�������������������������	�������Y�M�3�!��5�Z�s�������������	�y�m�U�F�B�G�m�y�����ɿ׿������ѿ��y�z�t�n�l�n�zÇÍÎÇ�z�z�z�z�z�z�z�z�z�zììàÓÎËÈÓÛàìùú��þùùììì�������������������Ŀѿٿ��޿����������a�X�Z�a�k�n�t�zÅ�z�v�n�a�a�a�a�a�a�a�a������������������������������������������������3�~���ֺ������⺗�~�L�3��ѿƿĿ��������ĿȿѿݿԿѿѿѿѿѿѿѿѻܻۻл������лܻ����@�Q�Y�T�@�+�����ĦġĢĦĮĳĿ������������ĿĴĳĦĦĦĦ�x�t�l�`�_�[�_�g�l�x�������������������xƬƧƝƦƳ�������0�4�3�8�5�)�������Ƭ�;�2�/�"���"�-�,�/�6�;�A�H�N�J�I�H�;�;�	�����Ӿʾ����׾��	�"�-�2�.�(�"��	��{�z������������������������Y�T�S�X�Y�b�f�q�r�����~�w�r�f�Y�Y�Y�Y�������������������Ƽü������������������T�O�H�B�>�C�H�M�T�`�a�m�m�m�k�f�a�[�T�T�Y�Q�M�Y�e�o�r�{�~�������������~�r�e�Y�YŭŬŇ�{�n�uŅŔŠŭŹ��������������Źŭ�Ŀ����������Ŀݿ��������������ѿ�E�E�E�E�E�F	FF1F=FJFcFhFdFVFJFBF4F$FE��	����������	������	�	�	�	�	�	�	�	�U�Q�I�C�<�7�:�<�I�U�b�n�s�{�|�{�o�n�b�U����ŭŠŎŅŇŎŠŹ��������������������`�T�H�9�3�1�;�H�a�z�����������������z�`�B�6�)����)�6�K�[�h�t�zāĄ�{�j�[�O�B�x�u�_�V�J�G�S�_�l�x�������������������x�������*�4�6�9�6�*�������������������������������+�"�������������������������������������������������ùóìëçàÞàìù��������üûùùùù�������������������ʼּݼּʼż����������������������������������
����
���#�$�#��
�
�
�
�
�
�
�
�
�
�������������������Ŀѿҿֿ׿տѿɿĿ����������޽�������(�3�4�<�4�(���������������������������������������������ù������ùϹܹ�������������ܹϹþ���޾��������	�������	��������������	���	�������������������������������������ʾ׾������߾׾ʾ����T�H�@�8�7�9�;�H�T�a�m�������������z�m�TDoDbDODVD^DoD|D�D�D�D�D�D�D�D�D�D�D�D�DoùèÛÓÇÀÇÓàìù����������������ù������$�0�1�6�9�0�$���������/�$�#�����#�/�<�H�L�U�b�b�a�U�H�<�/����¿�w�s�t¦¿����� �	�������ػû����������ûлܻ�����ܻлûûûý������������������������������������������������������������������������������ĿĻĳĦĤĚďċĦĳĿ����������������ĿE�E�E�E�E�E}E�E�E�E�E�E�E�E�E�E�E�E�E�E� [ 1 C   $ Q 7 W ) ? _ D ; * l l I 5 * � @ 1 i > A q 2 X C 4 9 ;  7 R . ; d B * $ 1 : L G Y ? p � x @ 9 F \ J ? D W * H T 9 X ^ ) F 0 R <    �  �  E  �    �  �  �  �  �  7  �  r  I  �  �  W  '  2    x  @  �  W    �    �  $  �  :  �  �    �  X  �  `  y  O  7    i  �  �  �  �  �  7  �  /  b  �  $  d  �  S  U  x  �  I  �  Q  f  �  3  �  �   ='�<�t�<�o�#�
%   ;D����`B��o�ě���`B�o�D���D���\)��9X��O߽49X�����h��9X��`B��t���"Ѽ�1�]/��`B�C��Y��C��P�`��/���'H�9�t���t����P�ixս'H�9��9X���罉7L��-�8Q콡���aG��aG��T���L�ͽaG�����e`B�T����C����-�e`B�m�h��������9X���P���"ѽ��㽮{�ȴ9���`�C�B3Br�B��B�BisB4�Bl�BYB��B�B�;B�
BO�B��A�ܚB&(TB+�B+Bd*B	�B��Bc)Bd�B�sB��B E)Bz�Bo^B��Bw�B��B#Z�B$%�B<�B"(>B�CB��BB�B0G�B�+B
�1B�2B�"B!:�B��B�-B��B
�RB,"�B.^B��BǶB��B>�B��B	��B��BM�B�IB��BS�B)�BTlB	ٗB)ŋB%~LB
X�B�B�MB?�B@�B�;B�B��B4�ABz�B<�B�sB��B�jB��BM�B:;A�A�B&�B*��BB�B@�B	:yB�Bd|B=,B�5B�B ��BW2BoB�ZB��B��B#NYB$?�B@B"0;B��BC{B�B0=}B�fB
��B��BŴB ��B̲B-�B�B=�B,=�B.yUB��B��BE[Bq�BA�B	��B� B?�B��B�qB�OB
ĊB?B	ǗB)�B%?�B
C&BB_B?�C�@A��A�tA���A�1�AN��A�%�A�H;A�-A͓�A�ף@�^KA�%�A��A���A��AquA�ٍA��GAv\sA�"�A!p�@gNAyY@@���A�j~@���BةA�AY֮AG�@�x�@��A��|?�d�A�MA|i;C�ѴA[}�A��}A��]A�/�A�wD@�TUA��4A���A��A�4C@�L=A�@A�-cAv�0A2�
A�Po>��2AY��AYs`AP�4A�TC�ɧA�$�B	υA�z�A��`@�D4A#	rA�A�l�C�C�?XA�5�A�M�AĖ�A��$AN��A�#A�~�A���A�~�A�_@�6�A��	A؈�A�hA��@AmgpA��$A��Ata�A�y�A!�@�;Ay��@��xA��@�.0B	1xA�roAY8AE�@ߨd@���A���?�6A�ZA|rC��LA[A���A���A���A�|7@�A���A���A�}rÁS@�)A1PA�mAuG�A5�+A�+;>���AY�AY�AN�~A��,C��A�rB
A��GA��@��A"RA��A�psC�             9   
                                 =   $   
               [      #                                  (   )            2   ,      /      &      
         	   %   
               	   !   :      	      &         	      "                     #      !   #               !   M   7         )         K      /         -      #                                 %   !            )                                          !            %                                                            !   E   %         !         A      +         +                                                      )                                          !            !               NX��N�e�NxxO�SN�+NQ{nO��LNx*�Og�OX��M�w�N�$�Nb:OY�,O�nP��PJN�4N��~O��N�:N1��P�DN@9�P�9N�'NıP#S�NFݡOv^�NWN���Nϼ!N�v�N��O�m
OEdxO�TNP-CO��OP��O͡�O�>�O�GN��O�K�N�հN�T�N��kNv�N!0'O��O/��N�/gO�O'��N9�OZ�O��O��O@�UN��lN�OO�>�N���N��N�2FO:�&O�  �  �  �    �  .  �  b    s  k  �  n  k  o  J  �  �  �  �  �  +  �  >  �  Q  �  �  �  �  �  �  b  �  =  �  4  �  [  2  �  L  c    O     R  y  C  �  .  @  �  ?  �  �    �  &  :  (  �  c  #  �  2    j  =D��=o<�t�;��
<t�<t��o���
�ě��#�
���
���
��`B�e`B�t��D���u�#�
�D���D����t���o��`B��C�������1��j��j��/���ě�������������h��P��w�t��t���P�q���0 Ž'u�,1�,1�8Q�<j�<j�<j�@��H�9�@��D���D���aG��L�ͽL�ͽ]/�y�#��C���+��hs���P������
��Q콸Q����������������������������
����������������������������������������������<BDN[`ggjig[NB<<<<<<��������������������������
 (2/*#
�����#/9;5/#!R[anz|������zna[ULLR���
#*/892#
������vz������zzvvvvvvvvvv&)+6>BJOOOOGB<63,)&&fhltw�����tlkhffffff#/3<GLH></%#
.19=QTZad__XQH;7/*-.����0e{������b<0���������������������������������������������������������������7BN[t���������tgNIB7<BOS[\[QOMB=<<<<<<<<pt�����tikpppppppppptz�������������ymiht),+)&�������������yz������������zyyyyy�������������������������������������~����������������������Xaht���������th`[WVX������������������������ 


������������

����������������������������������������������
�����������������������������'/<HMU[bfhdaURI</(%'%*/6CCIKC6*)%%%%%%%%��������������������st�����������tsnmns�����������������{z������ ��������������������������25BMNSSNLB?775222222ACOht}����th[QPLJCA��������		����������ntz������������tnnnn���������������������������������������������������#4HU[afhaUH</#���  " ����(/8<?Uaja\[VUHF</.+(LO[hssptzythg[TOKMJLY[gtw��������tgc[TSY��������������������agnpz���������zofaa]fnz�����������za]Y]#/<CGFD?/#)357<>51)(��������������������?BEN[^db\[TNB@@?????Z^``aht���������tg[Z��������������������#/0670#"qt����������utonqqqq��������������������6<<HKUaakha^USH<9466E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�¦²´¶¶²¯¦��������������������������������������H�/�(� �"�(�/�<�H�U�a�n�r�u�v�t�n�a�U�H�������������������������������������������������������ɾľʾ˾ʾ����������������s�j�Z�N�E�B�B�Z�g�s�y�����������������s�H�=�=�H�U�a�h�h�a�U�H�H�H�H�H�H�H�H�H�H�g�Z�T�O�U�Z�g�s���������������������s�gìæâßÝÝàâìù����������������ùì���������������������������������������Ҽr�j�f�]�f�i�r�|�����������������y�r�r�H�C�<�7�<�H�I�U�a�c�f�a�U�L�H�H�H�H�H�H�B�6�*�(�(�)�-�6�B�O�h�k�n�h�e�e�^�[�O�B�������s�Z�T�Z�f���������������������������������s�O�5�&�"�/�Z�s���������������`�Q�H�L�T�`�m�������Ŀٿ��ۿѿ����y�`�z�t�n�l�n�zÇÍÎÇ�z�z�z�z�z�z�z�z�z�zÓÐÌËÓÞàìóù��üù÷ìàÓÓÓÓ�������������������Ŀѿտ��ۿп��������a�Z�\�a�m�n�o�z�}�z�t�n�a�a�a�a�a�a�a�a���������������������������������������������"�e�����ֺ��
�����뺗�~�e�L��ѿƿĿ��������ĿȿѿݿԿѿѿѿѿѿѿѿѻ����ݻŻ������лܻ���@�O�M�?�%����ĦġĢĦĮĳĿ������������ĿĴĳĦĦĦĦ�l�b�_�^�_�l�x���������������������x�l�lƱƬƬƳ��������$�0�3�3�7�4�(�������Ʊ�/�+�"�!�"�/�;�H�J�H�D�;�/�/�/�/�/�/�/�/�����۾׾׾����	��!�(�,�&�"���	����{�z������������������������Y�T�S�X�Y�b�f�q�r�����~�w�r�f�Y�Y�Y�Y�������������������Ƽü������������������T�Q�H�D�>�C�H�O�T�]�a�l�j�e�a�Y�T�T�T�T�Y�Q�M�Y�e�o�r�{�~�������������~�r�e�Y�YŇ�{�q�xňŔŠŭŹ��������������ŹŭŠŇ�Ŀ��������ſѿݿ������������ݿ׿ѿ�E�E�E�E�E�F	FF1F=FJFcFhFdFVFJFBF4F$FE��	����������	������	�	�	�	�	�	�	�	�U�Q�I�C�<�7�:�<�I�U�b�n�s�{�|�{�o�n�b�UŠŠŖřšŭŹ��������������������ŹŭŠ�a�T�;�5�3�;�H�a�m�z�����������������z�a�B�6�)����)�6�K�[�h�t�zāĄ�{�j�[�O�B�x�q�l�a�_�Z�Z�_�l�x�}�����������������x�������*�4�6�9�6�*�������������������������������+�"�������������������������������������������������ùóìëçàÞàìù��������üûùùùù�������������������ʼּݼּʼż����������������������������������
����
���#�$�#��
�
�
�
�
�
�
�
�
�
�������������������ĿѿտտֿԿϿǿĿ����������޽�������(�3�4�<�4�(���������������������������������������������ù������ùϹܹ�������������ܹϹþ����������	��������	��������������	���	�������������������������������������ʾ׾������߾׾ʾ����T�H�A�:�8�;�;�H�T�a�m�z�����������z�m�TDoDbDODVD^DoD|D�D�D�D�D�D�D�D�D�D�D�D�Do��ùìàßÛàãìù��������������������������$�0�1�6�9�0�$���������/�&�#���#�/�<�H�K�U�`�]�U�H�<�/�/�/�/������¿¦�z�v�x¦¿��������������û����������ûлܻ�����ܻлûûûý������������������������������������������������������������������������������ĿĻĳĦĤĚďċĦĳĿ����������������ĿE�E�E�E�E�E}E�E�E�E�E�E�E�E�E�E�E�E�E�E� [ , C  $ Q , , # M _ D ; ' l a 8 5 % c E 1 d > B q 4 X : % 9 ;  < R 1 6 d B *  1 : @ G Y ? p � x @ 7 F \ J 7 D W 0 H 6 9 Q Y ) F 0 R <    �  &  E  T    �  h  {  �  �  7  �  r  �  �    s  '  �  �  S  @  �  W  �  �  �  Z  S  �  :  �  �    �    �  `  y  O  �  �  i  Z  �  �  �  �  7  �  /  2  �  $  d  k  S  U  .  �  �  �    �  �  3  �  �     A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  �  �  �  �  �  |  b  F  +    �  �  �  �  �  g  I  '    �  �  �  �  �  �  �  �  �  �  �  t  H    �  �  �  I  �  �  #  �  �  �  �  �  �  �  �  �  �    s  f  Z  N  A  5  )      �  �  �          �  �  �  �  j  !  �  i  �  P  H  L  %  �  �  �  ~  y  q  i  `  W  L  @  4  (        W  �  �  �  .  ;  H  V  Z  \  ^  `  c  e  g  i  k  m  n  p  r  v  y  |  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  ]  ?    �  �  v  �  �  �     )  O  a  a  _  [  Q  A  -    �  �  �  �  �  �  �  �  �  �      �  �  �  �  �  c  5    �  �  I    �  �  L  [  `  j  q  g    �    6  W  e  b  P  .  �  �  ]    k  c  [  S  K  C  ;  2  )          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  f  W  G  7  (  2  F  Y  n  f  ^  U  M  C  8  .  $        �  �  �  �  �  �  �  z  5  ?  L  ^  j  h  T  3    �  �  g  .  �  �  �    �  �  �  o  e  Z  M  ?  .      �  �  �  �  �  }  V  8  )    3  �  -  I  :    �  �  �  P    �  �  Z    �  e  	  �  5  �   �  �  �  �  �  �  �  �  �  z  H    �  �  �  �  �  \    �  3  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    e  @    �  �  j    �  X  �  �  �  �  �  x  d  V  I  H  B  4  #       �  �  �  �  q  ?  ^  l  x  �  �  �  z  r  j  `  V  K  @  .    �  A  �  �  J  +  *  *  )  )  (  (  '  '  &  $  !                	  X  �  �  �  �  _  +  �  �  W  �  k  �  d  �  d  �  �    %  >  6  /  '              �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  B    �  �  �  c  %  �  W   �  Q  G  =  .    �  �  �  �  �  a  A    �  �  �  �  n  K  (  �  �  �  �  �  �  �  �  v  f  S  @  +    �  �  �  �  {  X  �  �  �  �  �  �  �  �  �  u  X  5    �  �  K  �  \  �  �  >  U  l  y  �  �  �  �  �  �  s  \  A  "  �  �  �  �  W  *  #  m  �  �  �  �  �  �  �  �  {  q  ^  C    �  �  F  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  l  �  �  �  y  q  i  `  X  O  E  ;  1  '            �  �  b  a  ^  Z  U  K  =  )    �  �  �  �  b  9    �  �  E    �  �  �  �  x  b  H  *    �  �  �  {  V  :  !    �  =  B  =  8  3  )      �  �  �  �  �  �  �  �  �  e  G  #  �  �  �  �  �  �  �  �  �  t  8  �  �  t  .  �  �    i  �  �   i    +  4  3  %    �  �  �  �  �  V  %  �  �  g  �  z  �  V  �  �  �  �  u  n  m  y  p  X  =    �  �  �  h  ?    �  �  [  Q  F  ;  0      �  �  �  �  �  �  p  O  /  
   �   �   �  2  "    �  �  �  �  �  �  �  �  �  �  �  �  w  R    �  �  �    D  l  �  �  �  �  �  �  �  o  =  �  �  %  �  %  U   �  5  H  J  A  .    �  �  w  2  �  �  U     �  R    �    k  c  P  5    �  �  �  �  �  �  �  Y    �  �  K  ;  +  A  |  �  �  �  �  �  �  �        �  �  �  Q  �  n  �  J  {  7  O  I  C  <  6  0  *  !        �  �  �      7  P  i  �     �  �  �  �  �  �  �  }  \  9    �  �  S  �  �    x   �  R  K  D  9  .  #        �  �  �  �  �  �  �  �  �  x  P  y  n  c  U  G  +    �  �  �  �  R    �  �  d  +  �  �  y  C  7  ,  !    	  �  �  �  �  �  �  �  �  �  �  �  �  ~  `  �  �  }  r  g  \  R  G  =  2  &    
  �  �  �  �  �  �  �  .  )  $    	  �  �  �  �  �  �  �  j  P  6    �  �  �  �  >  @  6  "  �  �  �  Z    �  �  A  �  �  ;  �  [  �  `  �  �  �  �  �  �  �  �  �  �  �  �  u  [  <    �  �  �    g  ?  :  6  1  -  (  !          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  S  *  �    S  O  8    �  �  �  )  a  �  �  �  �  �  �  �  m  J  "  �  �  �  >  �  �    V  b    �  �  �  �  �  �  �  �  �  h  E  "  �  �  �  �  ]  3  	  �  �  �  |  k  ]  X  T  I  =  ,      �  �  �  �  	  !  9    "  &  "      �  �  �  o  ;    �  �  ?  �  �  5  �  Z  :  0    �  �  H  
�  
�  
-  	�  	E  �  *  �    L  f  6  �  �  �  �  �  (     	  �  �  z  ;  �  �  a    �  R    �  �  �  �  {  n  _  O  >  ,    
  �  �  �  �  �  �  �  �  �  �  �  K  Z  _  U  I  9       �  �  �  Z  -     �  �  p  C    �    "  #              �  �  �  o  *  �  �  9  �  �  j  �  �  �  �  �  �  �  �  �  y  l  `  T  E  1    	   �   �   �  2  (        �  �  �  �  �  �  �  �  �  �  �  x  n  e  [    �  �  �  �  �  �  �  t  Z  =    �  �  �  �  _  7  
  �  j  P  6    �  �  �  �  w  Q  *    �  �  �  P    �  �  >    
�  
�  
V  
  	�  	�  	P  �  �  .  �  [  �  Y  �  b  �  R  �