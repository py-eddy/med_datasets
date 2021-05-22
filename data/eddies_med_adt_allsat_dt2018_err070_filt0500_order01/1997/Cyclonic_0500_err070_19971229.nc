CDF       
      obs    @   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��+J        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P�YG        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �+   max       ;o        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?Tz�G�   max       @Fu\(�     
    �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?У�
=p    max       @v�          
   *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @Q�           �  4�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�)`            5,   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       � Ĝ   max       ��`B        6,   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�b   max       B2        7,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�^   max       B2        8,   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�>   max       C��        9,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >���   max       C��>        :,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          c        ;,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          Q        <,   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          A        =,   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P��        >,   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�t�j~��   max       ?�-w1��        ?,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �+   max       %           @,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?Tz�G�   max       @Fnz�G�     
   A,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?У�
=p    max       @v��z�H     
   K,   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @Q�           �  U,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��`            U�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Dy   max         Dy        V�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?������   max       ?�*0U2a|        W�      a         R   8   b               
         D   3      	            *            	   	         
   &   O   
                     !                     !         (               
         1      
         
   N�WP���N/p�N�N�P�'PV�CP�YGO"�O���Ou�~N��N�m^O+�O@�P��O�|?M��OԛN!
�OeN>ʹPƄO���N5XN��N,�ObVN��PO)�{ND-�O�p�P>��Ng$�Nօ}Oh�O��!N"bBN�Ne\OO|��N5N.�O��[NWh2O�'N�;�O'L�N�zN� �O�3�N���O)�cN���Nֈ�Nno�N�KNJ�
N�E�N�G�N�R�Ot��NԪ�NssPN�!�;o;o%   ��`B�#�
�#�
�49X�D���T���e`B�e`B�u��o��t����
��1��1��9X��9X��j��j�ě���/�������+�+�C��\)�t��',1�8Q�<j�P�`�P�`�Y��]/�]/�]/�]/�aG��aG��m�h��%��%��o��o��+��+��+��+��7L���T����置{��E��Ƨ�Ƨ��l����+)*123/)%�#<n�������nV<0����56BOSSOB965555555555�����������������������0Ic_JUSI#
������{������
��������|yx{Uz������	������z]SU}�����������������z}'6EO[hnh[QKB6)
EP[hkt~tsr}th[ZI;6E��������������������?BGN[_gc[VNB6:??????KP[bgt��~ytng_[PNIGKBCGO\hu���}ue\TOFCAB/<Ubijq����zna</~���������������{ww~��������������������ht������������~trjhhFHJUYZWUOHFCFFFFFFFF����������������������������������������HQTaz�������zmaGA=>H�����������������������������������������������������������������������������������������������>BEORZ[ghih[OB=;>>>>�����
#-/*#
�������������������������z��������������}wrqz��������������������)))+15,)&^anz�����zunaa_]^^^^egot���������tge^`ee��������������������"##)+)#!#0:<III<;0#!!!!!!ggt�����������tgggggz~��������������}wwz()6BGO[\[OB6)(((((((V[hjshh[YSVVVVVVVVVVZ]gt�����������tk[XZ6BOPWUOB>66666666666����"'������������������������./<GTUaia^WUH</+$$(.nqqnjaUHD@DHOUannnnn��������������������4Bt�����������vgNB44MNV[gkt|�tg[XNMMMMMMTUan{������{nb`XUQTz{{���������{zxxzzzz����

�����������������������������SUVaanwz�����zna\USS�����������������������

��������������������������������������������&*) �����&))5BLNRVYNB@50)&&&&�����������������������������uy����������ֺӺ̺պֺ�������������ֺֺֺ������N�<�(���,����������������������!����!�-�8�7�-�#�!�!�!�!�!�!�!�!�!�!������������������� ������������������н����u�p�z��������4�=����������пm�`�I�=�;�<�G�y���������ɿͿʿ��������mƳƘƁ�*����	�*�CƧ�������1�/�����Ƴ����������������������������������v�m�X�W�h�y���������¿ȿɿǿ����������������������������ûлۻۻ޻ܻлû��������������������ûлӻлƻû�����������������������������������������������������������������#�(�,�5�<�A�D�A�5������������������������þʾо׾޾��׾��M�3�/�4�A�M�s����������������������s�M�M�@�9�<�B�M�Y�f�r����������������f�Y�M�#�!��#�/�<�<�<�<�/�#�#�#�#�#�#�#�#�#�#����������������%�"�����������������������������������������������ž׾ʾ������z�s�f�]�^�f�s�����Ҿ������H�=�<�8�<�H�U�X�X�U�H�H�H�H�H�H�H�H�H�H�����������������������+�3�0��	�������@�:�C�H�L�R�r�~���������������~�r�e�Y�@�b�V�a�b�n�{Ł�{�y�n�b�b�b�b�b�b�b�b�b�bŠŖŘŠūŭŹŻ����������������ŹŭŠŠ�ܹԹϹʹϹչܹ������ܹܹܹܹܹܹܹܿ`�Z�T�G�;�.�"����"�.�;�G�T�V�`�k�j�`��׾׾Ծ׾�����������������������~�z�z�y�z�|�}�����������������������h�b�^�b�h�t�wāāā�u�t�h�h�h�h�h�h�h�hĸĳįĦĔĚĿ������#�'�'����������ĸ�)�(�3�>�B�L�V�e�r�����кغغ˺��~�e�@�)�"�����"�/�;�H�P�H�>�;�/�"�"�"�"�"�"�����������������������	�����������������I�F�=�4�1�=�@�I�V�b�n�o�s�s�p�o�b�V�I�I��������*�6�C�X�j�h�T�O�B�B�6�*��������������ûлӻл˻û����������������S�I�I�S�U�^�_�c�j�l�r�u�l�_�S�S�S�S�S�S�������������������������������������������������������������+�%�"��	�������й������������������������������	���'�'�(�'�����������������������������������&�*�!����������������������������������������������(�%��(�5�A�N�g�s���������������x�g�5�(��úùñõù����������������������������F$F"F'F1F=F>FGFJFOFVFcFeFiFeF[FVFJF=F1F$E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��t�q�h�h�`�`�d�h�t�z�|�|�w�}�t�t�t�t�t�tĳġČćčĜĦ��������������������Ŀĳ�	�����������	������	�	�	�	�	�	�	�û����������ûлܻ��������ܻ׻лü������'�4�@�M�O�M�A�@�4�'����D�D�D�D�D�D�D�D�D�EEEED�D�D�D�D�D�D����������������������������������������̿Ŀ������������������Ŀ̿ѿտٿֿѿȿĿ��n�n�k�n�q�z�z�}ÇÇÇÇ�}�z�n�n�n�n�n�nE*EEEEEE*E7ECELEPE\EPEMECE7E*E*E*E*ÇÄÇÈÐÓÓÞàâäàßÓÇÇÇÇÇÇ�`�_�V�S�L�H�S�`�l�y�������z�y�q�l�j�`�`�������������Ľн���������ݽнĽ���ÓÇÇ�}�|ÇËÓàìôîìäà×ÓÓÓÓ�������� �����&������������������������������������ùôðìììù������ I Q 8 % 3 : i A a 7 @ E c [ n ' 8 6 T S F - F M ` $ n : i 7 U F S , " . : V � 4 � < # C l = > " } i _ L c K J . [ : q J P G N 2  �    W  �  1  �  	  [  �  �  �  �  �  �  �    	  D  Z  �  h  r  �  N    A  �  �  �  l  k  d  �  �  N  �  7  �  �  �  �  ?  �  x  �  �  �  	    i  �  z  �  �  �    �  �  �      �  �  o�\��`B��C���j��+��;d��/�������ͼě���/�o��-��hs��/����/�0 ŽC���+�ixսC���P��w�'��T���49X������P�`�aG���������]/�ixսu�� Ž}�y�#���
�u��{���P�\��9X���P�����hs�����hs��vɽ�^5���ȴ9�	7L�����������   � ĜB�B&��BQ�B\�B%RdB+.B![B�B�QB0�B�BO"B	B2B�FBׁB"�5B
}�B|cB!&oB �nA�bB!KB�B��B7@B��B�DB��B/tB)�BI�B��B�B
 BӓB%$"B%�hB
HnBR�B�)B��B
��BZzB��B�kB��B�MB'B	XB	�B(M*B)24B��B
�`B�rB�<B�vBN�B�yB�B��B,TB
�RB�aB'?bB@�BTB%@ B*��B?�B �B@�B�B��Bv�B	>DB2B�fB�6B"�B
�eB��B!1�B �A�^B!K�B4'B��B>8BˀB��B��BJ�B ��B�B�_B�#B	ÒB@B$�*B%��B
�BA'B �B�)B
@@B|B�B�xB�LB<HB4�B�BB	HdB(|�B)>�B@B
��B��B�SB��B5�B%B?�B�zB@�B7�@H�+A���@orA��+A$��Am�TBӛA��AsJ.@�(@��A�!A�M,AMj�AA
G@���A�T�A1n%A��AK4KAĴ�A�4?�o$A�RfA���>�>Ac�xAWA�x�A��&A奀@
{�A���A���BƻB >;@�IZ@��}A��A���? q�?�XA���A��+A��fAΣ�C��C�!�A��OA�A[��@��?@��LC�5}B'Axu)A�NqC�� A�(lAqA(`�A�م@�A#A΂�@D�dA��@l�rA�{�A$�AmSB��A҃�At��@�9�@�`A�ĸA���AL��A=�^@��A�z�A1=�A�cAM��Aā�A�\}?�Q�A�A�d�>���Ad(AX(A� A���A��"@��A���A��B�GB B�@���@�fA��,A�kx?2�3?�&�A�{+A�z�A���A΅�C��>C�'�Aۍ�A��AZ�N@��@΍�C�(B3�Ax�@A��C��AʙuA9�A'r�Aʓ�@��AΉ�      b         S   9   c               
         D   3      	            +            	   	         
   &   P   
                     !   	                  !         )                        2      
                  G         A   1   Q      #                  /   !            #      %                           %   /            !                     !      !               )                                                A            %   ?                                       #      %                              -            !                     !      !                                                         N�jzP��N/p�N�N�O1�O�p�P��HO"�O2{N��YNz�N�m^O+�O�}O�V�OV�M��N�U/N!
�OeN>ʹPƄO���N5XN��N,�ObVN��PO)�{ND-�O�6�P;Ng$�N�n�Oh�O��!N"bBN"r�Ne\OOeAwN��N.�O��[NWh2O�'N�;�O��N�zN/FO�XN���O)�cN���Nֈ�Nno�N�KNJ�
NÝ�N�G�N�R�Ot��NԪ�NssPN�!�    �  �  �    �  0  �  x  K  y  E  �  �  �  �  �  Y  �  n  �  c      �  .  %  '  �  �  �  	�  �  �  �  �  �  |  �  C  �  Y  �  z  �  _  �  j  �  �  U  �  �  	W  �  �  �  ^  �  i  c  ,  1  	���o�o%   ��`B�q����h���D����1��`B�u�u��o��1�o�\)��1��j��9X��j��j�ě���/�������+�+�C��\)���,1�,1�<j�<j�P�`�P�`�]/�]/�e`B�aG��]/�aG��aG��m�h��%�����o��7L��\)��+��+��+��7L���T����罰 Ž�E��Ƨ�Ƨ��l����+')///)��#<{������mU<#���56BOSSOB965555555555��������������������#07<@@=6.&#��������������������gz���������������zhg}�����������������z}()6BOW[ONJEB6)JO[chhplh`[OGEJJJJJJ��������������������?BGN[_gc[VNB6:??????KP[bgt��~ytng_[PNIGKKO\chu����xuh\\OGEGK"/<HU\dd`daUH</#"}�����������������}}��������������������ot�����������ttmooooFHJUYZWUOHFCFFFFFFFF����������������������������������������HQTaz�������zmaGA=>H�����������������������������������������������������������������������������������������������>BEORZ[ghih[OB=;>>>>�����
#-/*#
�������������������������z��������������~xssz��������������������)))+15,)&_anz�����zqnga`^____egot���������tge^`ee��������������������"##)+)##,02<?<40#ggt�����������tgggggz��������������~zxxz))6BDOVOB6*)))))))))V[hjshh[YSVVVVVVVVVVZ]gt�����������tk[XZ6BOPWUOB>66666666666����"'������������������������//2<FHRU_[UH<:/,%%(/nqqnjaUHD@DHOUannnnn��������������������BN[t���������zgNB55BMNV[gkt|�tg[XNMMMMMMTUan{������{nb`XUQTz{{���������{zxxzzzz����

�����������������������������SUVaanwz�����zna\USS���������������������� 

 �������������������������������������������&*) �����&))5BLNRVYNB@50)&&&&�����������������������������uy�����������ۺֺκֺغ������������������������s�T�B�.�/�Z���������������������׻!����!�-�8�7�-�#�!�!�!�!�!�!�!�!�!�!������������������� ��������������������������������������Ľнݽ�ݽнĽ��������m�`�W�I�G�L�T�`�y����������������������ƱƁ�o�>�6�6�C�hƚƳ��������"���������������������������������������������������������������¿Ŀ��������������������ûлѻлϻɻû������������������������ûлѻлûû�����������������������������������������������������������������������#�(�,�5�<�A�D�A�5��������������������������žʾ˾Ծʾ������M�8�3�3�:�A�M�f�s���������������s�f�M�Y�S�M�G�C�M�Y�f�r�~������������v�r�f�Y�#�!��#�/�<�<�<�<�/�#�#�#�#�#�#�#�#�#�#��������������#����������������������������������������������������ž׾ʾ������z�s�f�]�^�f�s�����Ҿ������H�=�<�8�<�H�U�X�X�U�H�H�H�H�H�H�H�H�H�H�����������������������+�3�0��	�������@�:�C�H�L�R�r�~���������������~�r�e�Y�@�b�V�a�b�n�{Ł�{�y�n�b�b�b�b�b�b�b�b�b�bŠŖŘŠūŭŹŻ����������������ŹŭŠŠ�ܹԹϹʹϹչܹ������ܹܹܹܹܹܹܹܿ`�Z�T�G�;�.�"����"�.�;�G�T�V�`�k�j�`��׾׾Ծ׾�����������������������~�z�z�y�z�|�}�����������������������h�b�^�b�h�t�wāāā�u�t�h�h�h�h�h�h�h�hĻĶİĭĳ�����
��#�%�%���
��������Ļ�e�@�3�*�8�B�L�V�e�r�����Ϻ׺غ׺˺��~�e�"�����"�/�;�H�P�H�>�;�/�"�"�"�"�"�"����������������������������������������I�F�=�4�1�=�@�I�V�b�n�o�s�s�p�o�b�V�I�I��������*�6�C�X�j�h�T�O�B�B�6�*��������������ûлӻл˻û����������������_�W�S�P�S�]�_�b�l�n�r�l�_�_�_�_�_�_�_�_�����������������������������������������������������������
�#�"�!��	���������չ���������������������������	���'�'�(�'�����������������������������������&�*�!����������������������������������������������(�%��(�5�A�N�g�s���������������x�g�5�(��úùñõù����������������������������F$F#F$F)F1F4F=FBFJFVFcFgFdFcFZFVFJF=F1F$E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��h�c�b�f�h�t�w�y�y�t�h�h�h�h�h�h�h�h�h�hĤĝęĐĐĚĦĿ������������������ĿĳĤ�	�����������	������	�	�	�	�	�	�	�û����������ûлܻ��������ܻ׻лü������'�4�@�M�O�M�A�@�4�'����D�D�D�D�D�D�D�D�D�EEEED�D�D�D�D�D�D����������������������������������������̿Ŀ������������������Ŀ̿ѿտٿֿѿȿĿ��n�n�k�n�q�z�z�}ÇÇÇÇ�}�z�n�n�n�n�n�nE*E EEEE E*E7ECEKEPE[EPELECE7E*E*E*E*ÇÄÇÈÐÓÓÞàâäàßÓÇÇÇÇÇÇ�`�_�V�S�L�H�S�`�l�y�������z�y�q�l�j�`�`�������������Ľн���������ݽнĽ���ÓÇÇ�}�|ÇËÓàìôîìäà×ÓÓÓÓ�������� �����&������������������������������������ùôðìììù������ D N 8 % , ) j A U = > E c X   8 4 T S F - F M ` $ n : i 7 G A S ' " . : J � 2 x < # C l = > " b Y _ L c K J . [ 8 q J P G N 2  �  ;  W  �  ~  �  !  [  �  �  �  �  �  C  ~  �  	    Z  �  h  r  �  N    A  �  �  �  l  �  T  �  �  N  �  7  S  �  �  �  ?  �  x  �  �  M  	  s  �  �  z  �  �  �    �  �  �      �  �  �  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy  Dy                �  �  �  �  �  �  �  v  ]  C  %  �  �  9  �  �    S    �  �  R  A  !  �  �  �  2  �  0  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �    !  '  -  &      �  �  �  �  �  w  h  W  E  1      �  �  �  �  �  T    �    W  G  '  5  �  �  �  �  �    
    �  �  e  �      �  {  o  n  u  �  �  �  �  �  �  s  b  S  <    �  b  �  -  "  �  �  	  )  0    �  j  �  �  @  r  �  �  o    �  �  �  U  �  �  �  �  �  �  �  �  �  �  �  k  <    �  �  r  K  '      (  =  D  @  7  q  b  H  1      �  �  �  O  �  v  �      	       "  "  %  $  "       F    �  �  .  �  n  
  �  I  e  {  �  �  �  �  �  ~  y  q  C  �  �  �  >  �  �  _    E  5  &      �  �  �  �  �  �  n  O  /    �  �  �  o  A  �  �  t  c  O  :  %    �  �  �  �  �  �  �  p  C  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  `  6    �  i  �  �  V  �  �  �  �  �  �  ]    �  �  Z    �  �  �  :  �    >  `  �  �  �  �  �  �  �  Z  '  �  �  >  �  �  �  ]  �  �  �                     �  �  �  �  �  �  }  d  K  T  V  X  V  S  M  C  9  )      �  �  �  �  q  Z  I  C  =  �             �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  c  X  S  I  >  :  ,    �  �  �  �  �  �  d  4  �  �  �  �  �        �  �  �  �  �  �  }  T  *  �  �  �  R    �  }  c  a  U  D  /    �  �  �  j  -  �  �  l  ,  �  �    h  z            �  �  �  �  F    �  x  9    �  �  <  �  {        �  �  �  �  �  �  �  �  �  q  W  =  $    �  �  �  �  �  �  �  r  c  T  G  :  .  "    
  �  �  �  �  �  �  �  .  '       	  �  �  �  �  �  n  K  (    �  �  �  b  7    %  #         �  �  �  �  �  �  �  �  �  �  a  9    �  �  '          �  �  �  �  �  �  �  �  �  �  �  }  n  _  P  �  �  �  v  ^  E  +    �  �  �  �  w  :  �  i  �  x   �   t  �  �  �  �  �  q  \  C  )    �  �  �  z  Q  +    �  �  �  m  �  �  x  N  (    �  �  �  P  �  �  $  �  .  �  C  �  �  	�  	�  	�  	�  	�  	�  	�  	V  	  �  r    �    z  �  *  g  r    �  {  q  o  n  d  W  G  5       �  �  �  �  ^  4    
    �  �  �  �  �  �  �  �  �  �  w  d  O  8    �  �  �  t  6  �  x  i  Y  E  -    �  �  �  Y  !  �  �  S  
  �  �  Y  �  �  �  �  �  �  x  q  T  -  �  �  �  T    �  B  �  �  /   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  u  w  y  z  {  x  v  s  q  n  l  j  g  e  Z  M  @  3  &  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  E  �  /  A  A  6  "    �  �  �  j  >    �  �  c    L  �  �  x  �  �  �  �  �  �  �  �  �  �  �  �  �  z  i  W  D  0      Y  H  6  $    �  �  �  �  �  t  W  4    �  �  �  M    �  �  �  �  ~  q  b  V  G  1    �  �  �    L    �  �    z  z  q  h  ^  U  J  >  3  (        �  �  �  �  �  �  �  �  �  �  �  y  j  V  `  C     �  �  �  �  B  �  �  i    �  s  _  O  ?  .      �  �  �  �  �  h  I  %  �  �  �  \  !  �  �  �  �  �  �  �  �  m  ?  	  �  �  4  �  �  #  �  C  I  �  j  A  0    �  �  �  �  k  1  �  �  O  �  |  �  n  �  $  �  �  �  �  �  �  �  �  �  �  �  �  �  {  X  ,  �  �  �  �  Z  u  {  �  �  y  �  �  m  w  I    �  A  �  U  �  |    �  a  U  L  C  ;  1  &        �  �  �  �  �  m  Q  0    �  �  �  �  �  q  c  U  H  ;  /  !      �  �  �  �  �  �  �  u  �  �  �  �  �  �  r  d  T  B  /      �  �  �  �  t  T  5  	W  	&  �  �  u  /  �  �  S  
  �  �  A  �  �  )  �      �  �  r  b  N  <  -      �  �  �  �  �  �  �    j  T  a  q  �  �  z  W  /  �  �  �  V    �  �  P    �  w  .  �  C  �  �  �  V    �  h  
  �  D  �  o    �  >  �  �  0  �  y    \  X  C  !  �  �  G  �  d  �  s  �  
�  
  	       �  �  V  �  �  �  �  q  U  8    �  �  �  y  G    �  �  W  �    J  i  K  ,  	  �  �  �  �  x  f  X  J  ?  2  $    �  �  �  �  c  b  ^  S  G  ;  4  1  0  *  "      �  �  t  '  �  �  P  ,  	  �  �  �  i  8    �  �  c  )  �  �  _    �  Z  �  u  1  ,  '  &  %  %  $  "      �  �  �  �  p  S  5    �  �  	�  	�  	�  	W  	  �    4  �  �  7  �  q  �  p  �  J    �  �